'use strict';
const Mustache = require('mustache');
const lodash = require('lodash');
const changeCase = require('change-case');
const async = require('async');
const parser = require('./shape');
const utils = require('./utils');
const types = require('./types');
const template = require('./template');
const fs = require('fs');
const path = require('path');


/**
 * Define template variables.
 */
const nameVar       = '{{name}}';
const tensorInVar   = '{{name}}_in';
const tensorOutVar  = '{{name}}_out';
const dataInVar     = '{{name}}_in_data';
const dataOutVar    = '{{name}}_out_data';
const cvVar         = '{{name}}_cv';
const dimVar        = '{{name}}_dims_';
const dimSizeInVar  = '{{name}}_in_dims_sz_';
const dimSizeOutVar = '{{name}}_out_dims_sz_';
const cvShapeVar    = '{{name}}_cv_shape';


function ascendingId(a, b) {
  return a.id - b.id;
}

function lowerAndSnake(str) {
  return changeCase.lowerCase(changeCase.snakeCase(str));
}

function capitalAndCamel(str) {
  return changeCase.upperCaseFirst(changeCase.camelCase(str));
}

const registerOpInput = function(opsMeta) {
  let result = [];
  if (opsMeta && opsMeta.inputs) {
    result = result.concat(opsMeta.inputs);
  }
  if (opsMeta && opsMeta.inputoutputs) {
    result = result.concat(opsMeta.inputoutputs.map((obj) => {
      return Object.assign({}, obj, { name: `${obj.name}_in` });
    }));
  }
  result.sort(ascendingId);
  return result.map((obj) => {
    let type = undefined;
    if (obj.typeFormat === 'cv') {
      type = types.cvToTf(obj.dtype);
    } else if (obj.typeFormat === 'std') {
      type = types.stdToTf(obj.dtype);
    }
    return { name: obj.name, type };
  });
};

const registerOpInputFn = function() {
  return `.Input("${lowerAndSnake(this.name)}: ${this.type}")`;
};

const registerOpOutput = function(opsMeta) {
  let result = [];
  if (opsMeta && opsMeta.outputs) {
    result = result.concat(opsMeta.outputs);
  }
  if (opsMeta && opsMeta.inputoutputs) {
    result = result.concat(opsMeta.inputoutputs);
  }
  result.sort(ascendingId);
  return result.map((obj) => {
    let type = undefined;
    if (obj.typeFormat === 'cv') {
      type = types.cvToTf(obj.dtype);
    } else if (obj.typeFormat === 'std') {
      type = types.stdToTf(obj.dtype);
    }
    return { name: obj.name, type };
  });
};

const registerOpOutputFn = function() {
  return `.Output("${lowerAndSnake(this.name)}: ${this.type}")`;
};

const opAttributes = function(opsMeta) {
  let result = [];
  if (opsMeta && opsMeta.attributes) {
    result = result.concat(opsMeta.attributes);
  }
  result.sort(ascendingId);
  return result.map((obj) => {
    return { name: obj.name, type: obj.type, defaultVal: obj.defaultVal };
  });
};

const registerOpAttrFn = function() {
  const type = this.defaultVal ? `${this.type} = ${this.defaultVal}` : `${this.type}`;
  return `.Attr("${lowerAndSnake(this.name)}: ${type}")`;
};

const getAttributesFn = function() {
  return `OP_REQUIRES_OK(context, context->GetAttr("${lowerAndSnake(this.name)}", &${this.name}_));`;
};

const declareAttributesFn = function() {
  return `${this.type}      ${this.name}_;`;
}

const registerOpShape = function(opsMeta) {
  let result = [];
  if (opsMeta && opsMeta.outputs) {
    result = result.concat(opsMeta.outputs);
  }
  if (opsMeta && opsMeta.inputoutputs) {
    result = result.concat(opsMeta.inputoutputs);
  }
  result.sort(ascendingId);
  return result.map((obj, index) => {
    return { regIdx: index, pShape: obj.pShape };
  });
};

const registerOpShapeFn = function() {
  const tfRank = this.pShape.tfRank;
  let shapeStr;
  if (tfRank === 0) {
    shapeStr = `c->Scalar()`;
  }
  else if (tfRank === 1) {
    shapeStr = `c->Vector(${getDim(this.pShape.dimDtorArr[0])})`;
  }
  else if (tfRank === 2) {
    // There two possible different shapes can have the same tfrank (>=2), which are:
    //  1. Two dimensional descriptors with a single channel data descriptor, eg. [ none, none, CV_64F ]
    //  2. One dimentional descriptor with a multichannel data descriptor, eg. [ none, CV_64FC3 ]
    // We need to figure out which one is the case.
    let secDim;
    if (this.pShape.dataDtor.channels === 1) {
      secDim = getDim(this.pShape.dimDtorArr[1]);
    } else {
      secDim = this.pShape.dataDtor.channels;
    }
    shapeStr = `c->Matrix(${getDim(this.pShape.dimDtorArr[0])}, ${secDim})`;
  }
  else if (tfRank >= 3) {
    const argus = this.pShape.dimDtorArr.map((elem) => {
      return getDim(elem);
    }).join(', ');
    if (this.pShape.dataDtor.channels === 1) {
      shapeStr = `c->MakeShape({ ${argus} })`;
    } else {
      shapeStr = `c->MakeShape({ ${argus}, ${this.pShape.dataDtor.channels} })`;
    }
  }
  return `c->set_output(${this.regIdx}, ${shapeStr});`;

  function getDim(dimDtor) {
    return dimDtor.dims === 'none' ? 'InferenceContext::kUnknownDim' : parseInt(dimDtor.dims);
  }
};

const computeInput = function(opsMeta) {
  let result = [];
  if (opsMeta && opsMeta.inputs) {
    result = result.concat(opsMeta.inputs);
  }
  if (opsMeta && opsMeta.inputoutputs) {
    result = result.concat(opsMeta.inputoutputs);
  }
  result.sort(ascendingId);
  return result.map((obj, index) => {
    return { regIdx: index, name: obj.name, shape: obj.shape, pShape: obj.pShape };
  });
};

const computeInputFn = function() {
  const template = `
  const Tensor& {{tensorInVar}} = context->input({{regIdx}});
  OP_REQUIRES(context, {{tensorInVar}}.dims() == {{tensorInRank}},
              errors::InvalidArgument("{{name}} must be {{tensorInRank}}-dimensional",
              {{tensorInVar}}.shape().DebugString()));
  auto {{dataInVar}} = {{tensorInVar}}.tensor<{{tensorDtype}}, {{tensorInRank}}>();
  {{{decTfDimSize}}}
  {{{cvtTensorToCv}}}
  `;
  const view = {
    name: this.name,
    regIdx: this.regIdx,
    tensorInRank: this.pShape.tfRank,
    tensorInVar: Mustache.render(tensorInVar, { name: this.name }),
    dataInVar: Mustache.render(dataInVar, { name: this.name }),
    tensorDtype: this.pShape.dataDtor.format === 'cv' ? types.cvToStd(this.pShape.dataDtor.dtype) :
                                                        this.pShape.dataDtor.dtype,
    decTfDimSize: Mustache.render(declareTfDimSize(this.pShape.tfRank), { name: this.name }),
    cvtTensorToCv: Mustache.render(convertTensorToCvByShape(this.pShape), { name: this.name })
  }
  return Mustache.render(template, view);

  function declareTfDimSize(rank) {
    return lodash.range(rank).map((idx) => {
      return `const int32 ${dimSizeInVar}${idx} = static_cast<int32>(${tensorInVar}.dim_size(${idx}));`;
    }).join('\n');
  }

  /**
   * Generates convertion code, given the parsed CV shape object, for copying data from a tensor to a
   * CV data structure.
   *
   * @param {array} pShape The parsed shape object specifiying an n-dimentional CV array shape.
   * @returns {string} Returns the convertion code string.
   * @example
   *
   */
  function convertTensorToCvByShape(pShape) {
    const dataDtor = pShape.dataDtor;
    const cvRank = pShape.cvRank;

    let declareStr, loopStr;
    switch (pShape.type) {
      case parser.SCALAR:
        declareStr = `${pShape.varDecStr} ${cvVar};`;
        loopStr = `${cvVar} = ${dataInVar}(0);`;
        break;
      case parser.VEC_OF_PRIM:
        declareStr = `${pShape.varDecStr} ${cvVar};`;
        loopStr = `${loopVecOfPrim(0, cvRank)}`;
        break;
      case parser.VEC_OF_MAT:
        if (pShape.dataDtor.ctype === 'Matx' ||
            pShape.dataDtor.ctype === 'Vec') {
          declareStr = `${pShape.varDecStr} ${cvVar};`;
        } else {
          // default is Mat
          declareStr = `const int ${cvShapeVar}[] = { ${utils.expandArgus(dimSizeInVar, pShape.vecDims, cvRank)} };
          ${pShape.varDecStr} ${cvVar};`;
        }
        loopStr = `${loopVecOfMat(0, pShape.vecDims, pShape.matDims, pShape.varMatDecStr, dataDtor)}`;
        break;
      case parser.MAT:
        const matDtype = dataDtor.format === 'cv' ? dataDtor.toString() : types.stdToCv(dataDtor.toString());
        if (pShape.dataDtor.ctype === 'Matx' ||
            pShape.dataDtor.ctype === 'Vec') {
          declareStr = `${pShape.varDecStr} ${cvVar};`;
        } else {
          declareStr = `const int ${cvShapeVar}[] = { ${utils.expandArgus(dimSizeInVar, cvRank)} };
          ${pShape.varDecStr} ${cvVar}(${cvRank}, ${cvShapeVar}, ${matDtype});`;
        }
        loopStr = `${loopMat(0, cvRank, dataDtor)}`;
        break;
    }

    return `${declareStr}
    ${loopStr}`;

    /**
     * Generates a nested loops for copying data from a tensor to a vector.
     *
     * @param {number} start The start index of the outermost loop.
     * @param {number} end The end index of the innermost loop.
     * @returns {string} Returns the nested loops string.
     * @example
     *
     * // input shape: [ vector:none, vector:none, int ]
     * const dataDtor = { type: 'int', channels: 1 };
     * loopVecOfPrim(0, 2, dataDtor);
     * // =>
     * // a_cv.resize(a_dims_size_0);
     * // for (int a_dims_0 = 0; a_dims_0 < a_dims_size_0; a_dims_0++) {
     * //   a_cv[a_dims_0].resize(a_dims_size_1);
     * //   for (int a_dims_1 = 0; a_dims_1 < a_dims_size_1; a_dims_1++) {
     * //     if (isnan(a_data(a_dims_0, a_dims_1))) { break; }
     * //     a_cv[a_dims_0].push_back(a_data(a_dims_0, a_dims_1));
     * //   }
     * // }
     *
     * // input shape: [ vector:none, int ]
     * dataDtor = { type: 'int', channels: 1 };
     * loopVecOfPrim(0, 1);
     * // =>
     * // a_cv.resize(a_dims_size_0);
     * // for (int a_dims_0 = 0; a_dims_0 < a_dims_size_0; a_dims_0++) {
     * //   a_cv.push_back(a_data(a_dims_0));
     * // }
     */
    function loopVecOfPrim(start, end) {
      if (start === end) {
        let guardStr = '';
        if (end > 1) {
          guardStr = `if (isnan(${dataInVar}(${utils.expandArgus(dimVar, end)}))) { break; }`;
        }
        const data = `${dataInVar}(${utils.expandArgus(dimVar, end)})`;
        return `${guardStr}
        ${cvVar}${utils.expandArrayAccessor(dimVar, end - 1)}.push_back(${data});`;
      }

      return `${resizeVector(start)}
      for (int ${dimVar + start} = 0; ${dimVar + start} < ${dimSizeInVar + start}; ${dimVar + start}++) {
        ${loopVecOfPrim(++start, end)}
      }`;
    }

    /**
     * Generates a nested loops for copying data from a tensor to a vector of Mat object.
     *
     * @param {number} start The start index of the outermost loop.
     * @param {number} vecDims The number of dimensions of the vector part in the input shape.
     * @param {number} matDims The number of dimensions of the Mat part in the input shape.
     * @param {string} varMatDecStr The declare string of the Mat object.
     * @param {object} dataDtor The data descriptor that used to construct the innermost loop body.
     * @returns {string} Returns the nested loops string.
     * @example
     *
     * // input shape: [ vector:none, none, none, CV_64FC3 ]
     * //
     * const vecDims = 1;
     * const matDims = 2;
     * const dataDtor = { type: '64F', channels: 3 };
     * loopVecOfMat(0, vecDims, matDims, dataDtor);
     * // =>
     * // a_cv.resize(a_dims_size_0);
     * // for (int a_dims_0 = 0; a_dims_0 < a_dims_size_0; a_dims_0++) {
     * //   Mat mat(2, mat_shape, CV_64FC3);
     * //   for (int a_dims_1 = 0; a_dims_1 < a_dims_size_1; a_dims_1++) {
     * //     for (int a_dims_2 = 0; a_dims_2 < a_dims_size_2; a_dims_2++) {
     * //       mat.at<Vec<double, 3>>(a_dims_1, a_dims_2)[0] = a_data(a_dims_0, a_dims_1, a_dims_2, 0);
     * //       mat.at<Vec<double, 3>>(a_dims_1, a_dims_2)[1] = a_data(a_dims_0, a_dims_1, a_dims_2, 1);
     * //       mat.at<Vec<double, 3>>(a_dims_1, a_dims_2)[2] = a_data(a_dims_0, a_dims_1, a_dims_2, 2);
     * //     }
     * //   }
     * //   a_cv.push_back(mat);
     * // }
     */
    function loopVecOfMat(start, vecDims, matDims, varMatDecStr, dataDtor) {
      if (start === vecDims) {
        let guardStr = '';
        if (vecDims > 1) {
          const paddingZeros = matDims + (dataDtor.channels > 1 ? 1 : 0);
          guardStr = `if (isnan(${dataInVar}(${utils.expandArgus(dimVar, vecDims) + ', 0'.repeat(paddingZeros)}))) { break; }`;
        }
        const tmpVar = 'mat';
        const tmpVarMatDecArguStr = dataDtor.ctype === 'Mat' ? `(${matDims}, ${cvShapeVar}, ${dataDtor.toString()})` : '';
        const tmpVarMatDecStr = `${varMatDecStr} ${tmpVar}${tmpVarMatDecArguStr}`;
        return `${guardStr}
        ${tmpVarMatDecStr};
        ${loopMat(vecDims, matDims + vecDims, dataDtor, vecDims, tmpVar)}
        ${cvVar}${utils.expandArrayAccessor(dimVar, vecDims - 1)}.push_back(${tmpVar});`;
      }

      return `${resizeVector(start)}
      for (int ${dimVar + start} = 0; ${dimVar + start} < ${dimSizeInVar + start}; ${dimVar + start}++) {
        ${loopVecOfMat(++start, vecDims, matDims, varMatDecStr, dataDtor)}
      }`;
    }

    /**
     * Generates a nested loops for copying data from a tensor to a Mat object.
     *
     * @param {number} start The start index of the outermost loop.
     * @param {number} end The end index of the innermost loop.
     * @param {object} dataDtor The data descriptor that used to construct the innermost loop body.
     * @param {number} [cvStartOffset=0] The start data index offset of the Mat object when copying the data.
     * @param {string} [newCvVar=""] Using the specified CV variable name instead of the global name.
     * @returns {string} Returns the nested loops string.
     * @example
     *
     * // input shape: [ none, none, CV_64FC3 ]
     * //
     * // Prerequisite declaration:
     * //
     * //   const int a_shape[] = { a_dims_size_0, a_dims_size_1 };
     * //   Mat a_cv(2, a_shape, CV_64FC3);
     * const dataDtor = { type: '64F', channels: 3 };
     * loopMat(0, 2, dataDtor);
     * // =>
     * // for (int a_dims_0 = 0; a_dims_0 < a_dims_size_0; a_dims_0++) {
     * //   for (int a_dims_1 = 0; a_dims_1 < a_dims_size_1; a_dims_1++) {
     * //     a_cv.at<Vec<double, 3>>(a_dims_0, a_dims_1)[0] = a_data(a_dims_0, a_dims_1, 0);
     * //     a_cv.at<Vec<double, 3>>(a_dims_0, a_dims_1)[1] = a_data(a_dims_0, a_dims_1, 1);
     * //     a_cv.at<Vec<double, 3>>(a_dims_0, a_dims_1)[2] = a_data(a_dims_0, a_dims_1, 2);
     * //   }
     * // }
     *
     * // input shape: [ none, none, CV_64F ]
     * const dataDtor = { type: '64F', channels: 1 };
     * loopMat(0, 2, dataDtor);
     * // =>
     * // for (int a_dims_0 = 0; a_dims_0 < a_dims_size_0; a_dims_0++) {
     * //   for (int a_dims_1 = 0; a_dims_1 < a_dims_size_1; a_dims_1++) {
     * //     a_cv.at<double>(a_dims_0, a_dims_1) = a_data(a_dims_0, a_dims_1);
     * //   }
     * // }
     */
    function loopMat(start, end, dataDtor, cvStartOffset, newCvVar) {
      if (start === end) {
        const cvArgus = utils.expandArgus(`${dimVar}`, cvStartOffset, end);
        const dataArgus = utils.expandArgus(`${dimVar}`, end);
        if (dataDtor.channels === 1) {
          return `${newCvVar ? newCvVar : cvVar}${dataDtor.accessor(cvArgus)} = ${dataInVar}(${dataArgus});`;
        } else if (dataDtor.channels > 1) {
          return lodash.range(dataDtor.channels).map((channelIdx) => {
            return `${newCvVar ? newCvVar : cvVar}${dataDtor.accessor(cvArgus)}[${channelIdx}] = ${dataInVar}(${dataArgus}, ${channelIdx});`;
          }).join('\n');
        } else {
          throw new Error(`Invalid channel number: ${dataDtor.channels}`);
        }
      }

      return `for (int ${dimVar + start} = 0; ${dimVar + start} < ${dimSizeInVar + start}; ${dimVar + start}++) {
        ${loopMat(++start, end, dataDtor, cvStartOffset, newCvVar)}
      }`;
    }

    /**
     * Generates a statement that resize a specific layer of multi-layer vector.
     *
     * @param {number} layer The layer index of the vector.
     * @returns {string} Returns the resized vector string.
     * @example
     *
     * resizeVector(2);
     * // =>
     * // a_cv[a_dims_0][a_dims_1].resize(a_dims_size_2);
     */
    function resizeVector(layer) {
      return `${cvVar}${utils.expandArrayAccessor(dimVar, layer)}.resize(${dimSizeInVar + layer});`;
    }
  }
};

const computeExecute = function(opsMeta) {
  let result = {};
  result.fnName = opsMeta.fnName;
  if (opsMeta.inputs) {
    result.inputs = opsMeta.inputs.map((obj) => {
      return { id: obj.id, name: obj.name };
    });
  }
  if (opsMeta.outputs) {
    result.outputs = opsMeta.outputs.map((obj) => {
      return { id: obj.id, name: obj.name, pShape: obj.pShape };
    });
  }
  if (opsMeta.inputoutputs) {
    result.inputoutputs = opsMeta.inputoutputs.map((obj) => {
      return { id: obj.id, name: obj.name };
    });
  }
  if (opsMeta.attributes) {
    result.attributes = opsMeta.attributes.map((obj) => {
      return { id: obj.id, name: obj.name };
    });
  }
  return result;
};

const computeExecuteFn = function() {
  let declareOutputStr = '';
  let argus = [];

  if (this.inputs) { argus = argus.concat(addPostfixToName(this.inputs, '_cv')); }
  if (this.outputs) {
    const outputs = addPostfixToName(this.outputs, '_cv');
    argus = argus.concat(outputs);
    outputs.forEach((item) => {
      declareOutputStr += `${item.pShape.varDecStr} ${item.name};\n`;
    });
  }
  if (this.inputoutputs) {
    // Since the CV variables for inputoutput has been declared in input section, so don't
    // declare CV variables again.
    argus = argus.concat(addPostfixToName(this.inputoutputs, '_cv'));
  }
  if (this.attributes) { argus = argus.concat(addPostfixToName(this.attributes, '_')); }

  argus.sort(ascendingId);

  return `${declareOutputStr}
  ${this.fnName}(${argus.map((item) => { return item.name; }).join(', ')});`;

  function addPostfixToName(arr, postfix) {
    return arr.map((item) => {
      item.name = `${item.name}${postfix}`;
      return item;
    });
  }
};

const computeOutput = function(opsMeta) {
  let result = [];
  if (opsMeta && opsMeta.outputs) {
    result = result.concat(opsMeta.outputs);
  }
  if (opsMeta && opsMeta.inputoutputs) {
    result = result.concat(opsMeta.inputoutputs);
  }
  result.sort(ascendingId);
  return result.map((obj, index) => {
    return { regIdx: index, name: obj.name, shape: obj.shape, pShape: obj.pShape };
  });
};

const computeOutputFn = function() {
  const tfShapeArgus = utils.expandArgus((idx) => { return `${dimSizeOutVar + idx}`; }, this.pShape.tfRank);

  const template = `
  {{{decCvDimSize}}}
  Tensor *{{tensorOutVar}};
  OP_REQUIRES_OK(context, context->allocate_output({{regIdx}}, TensorShape({ {{tfShapeArgus}} }), &{{tensorOutVar}}));
  auto {{dataOutVar}} = {{tensorOutVar}}->tensor<{{tensorDtype}}, {{tensorOutRank}}>();
  {{{cvtCvToTensor}}}
  `;
  const view = {
    name: this.name,
    regIdx: this.regIdx,
    tensorOutRank: this.pShape.tfRank,
    tensorOutVar: Mustache.render(tensorOutVar, { name: this.name }),
    dataOutVar: Mustache.render(dataOutVar, { name: this.name }),
    tfShapeArgus: Mustache.render(tfShapeArgus, { name: this.name }),
    tensorDtype: this.pShape.dataDtor.format === 'cv' ? types.cvToStd(this.pShape.dataDtor.dtype) :
                                                        this.pShape.dataDtor.dtype,
    decCvDimSize: Mustache.render(declareCvDimSize(this.pShape), { name: this.name }),
    cvtCvToTensor: Mustache.render(convertCvToTensorByShape(this.pShape), { name: this.name })
  };
  const computeOutputStr = Mustache.render(template, view);

  if (this.pShape.vecDims > 0) {
    return utils.genIfElse(
      Mustache.render(`${cvVar}.size() == 0`, { name: this.name }),
      `OP_REQUIRES_OK(context, context->allocate_output(${this.regIdx}, TensorShape({ 0 }), NULL));`,
      computeOutputStr
    );
  } else {
    return computeOutputStr
  }

  /**
   * Declares dimension size variables.
   *
   * Example:
   *
   *  [ vector:none, 3, 3, CV_64FC2 ]
   *  vecDims: 1
   *  matDims: 2
   *
   *  const int32 a_dims_size_0 = static_cast<int32>(a_cv.size());
   *  const int32 a_dims_size_1 = static_cast<int32>(a_cv[0].size[0]);
   *  const int32 a_dims_size_2 = static_cast<int32>(a_cv[0].size[1]);
   *  const int32 a_dims_size_3 = static_cast<int32>(a_cv[0].channels());
   *
   * XXX: Note: If the shape type is vector of something, we treat all the vectors as fixed
   *            length, so we only check the size of the first element for each vector layer.
   */
  function declareCvDimSize(pShape) {
    const vecDims = pShape.vecDims;
    const matDims = pShape.matDims;
    let outStr = '', idx = 0, i;

    for (i = 0; i < vecDims; i++) {
      outStr += `const int32 ${dimSizeOutVar + idx} = static_cast<int32>(${cvVar}${expand(i)}.size());\n`;
      idx++;
    }

    if (pShape.dataDtor.ctype === 'Mat') {
      for (i = 0; i < matDims; i++) {
        outStr += `const int32 ${dimSizeOutVar + idx} = static_cast<int32>(${cvVar}${expand(vecDims)}.size[${i}]);\n`;
        idx++;
      }
    } else {
      for (i = 0; i < matDims; i++) {
        outStr += `const int32 ${dimSizeOutVar + idx} = ${pShape.matDimDtorArr[i].dims};\n`;
        idx++;
      }
    }

    if (pShape.dataDtor.channels > 1) {
      outStr += `const int32 ${dimSizeOutVar + idx} = static_cast<int32>(${cvVar}${expand(vecDims)}.channels());\n`;
    }

    return outStr;

    function expand(length) {
      return lodash.range(length).map(() => { return '[0]'; }).join('');
    }
  }

  /**
   * Generates convertion code, given the parsed CV shape object, for copying data from a CV data
   * structure to a tensor.
   *
   * @param {array} pShape The parsed shape object specifiying an n-dimentional CV array shape.
   * @returns {string} Returns the convertion code string.
   * @example
   *
   */
  function convertCvToTensorByShape(pShape) {
    const dataDtor = pShape.dataDtor;
    const cvRank = pShape.cvRank;

    let loopStr;
    switch (pShape.type) {
      case parser.SCALAR:
        loopStr = `${dataOutVar}(0) = ${cvVar}`;
        break;
      case parser.VEC_OF_PRIM:
        loopStr = `${loopVecOfPrim(0, cvRank)}`;
        break;
      case parser.VEC_OF_MAT:
        loopStr = `${loopVecOfMat(0, pShape.cvRank, pShape.vecDims, dataDtor)}`;
        break;
      case parser.MAT:
        loopStr = `${loopMat(0, cvRank, dataDtor)}`;
        break;
    }

    return `${loopStr}`;

    function loopVecOfPrim(start, end) {
      if (start === end) {
        return `${dataOutVar}(${utils.expandArgus(dimVar, end)}) = ${cvVar}${utils.expandArrayAccessor(dimVar, end)};`;
      }

      return `for (int ${dimVar + start} = 0; ${dimVar + start} < ${dimSizeOutVar + start}; ${dimVar + start}++) {
      ${loopVecOfPrim(++start, end)}
      }`;
    }

    /**
     * Generates a nested loops for copying data from a CV object to a tensor.
     *
     * @param {number} start The start index of the outermost loop.
     * @param {number} end The end index of the innermost loop.
     * @param {number} vecDims Number of the vector dimensions.
     * @param {object} dataDtor The data descriptor that used to construct the innermost loop body.
     * @returns {string} Returns the nested loops string.
     * @example
     *
     * // output shape: [ 4, CV_64FC3 ]
     * const dataDtor = { type: 'double', channels: 3 };
     * loop(0, 1, dataDtor);
     * // =>
     * // for (int a_dims_0 = 0; a_dims_0 < a_cv.size[0]; a_dims_0++) {
     * //   a_data(a_dims_0, 0) = r_cv.at<double>(a_dims_0, 0);
     * //   a_data(a_dims_0, 1) = r_cv.at<double>(a_dims_0, 1);
     * //   a_data(a_dims_0, 2) = r_cv.at<double>(a_dims_0, 2);
     * // }
     *
     * // output shape: [ 3, 3, double ]
     * const dataDtor = { type: 'double', channels: 1 };
     * loop(0, 2, dataDtor);
     * // =>
     * // for (int a_dims_0 = 0; a_dims_0 < a_cv.size[0]; a_dims_0++) {
     * //   for (int a_dims_1 = 0; a_dims_1 < a_cv.size[1]; a_dims_1++) {
     * //     a_data(a_dims_0, a_dims_1) = r_cv.at<double>(a_dims_0, a_dims_1);
     * //   }
     * // }
     *
     * // output shape: [ 3, 3, CV_64FC2 ]
     * const dataDtor = { type: 'double', channels: 2 };
     * loop(0, 2, dataDtor);
     * // =>
     * // for (int a_dims_0 = 0; a_dims_0 < a_cv.size[0]; a_dims_0++) {
     * //   for (int a_dims_1 = 0; a_dims_1 < a_cv.size[1]; a_dims_1++) {
     * //     a_data(a_dims_0, a_dims_1, 0) = r_cv.at<double>(a_dims_0, a_dims_1, 0);
     * //     a_data(a_dims_0, a_dims_1, 1) = r_cv.at<double>(a_dims_0, a_dims_1, 1);
     * //   }
     * // }
     */
    function loopVecOfMat(start, end, vecDims, dataDtor) {
      if (start === end) {
        //const dtype = dataDtor.format === 'cv' ? types.cvToStd(dataDtor.dtype) : dataDtor.dtype;
        if (dataDtor.channels === 1) {
          return `${dataOutVar}(${utils.expandArgus(dimVar, end)}) = ${cvVar}${utils.expandArrayAccessor(dimVar, vecDims)}${dataDtor.accessor(utils.expandArgus(dimVar, vecDims, end))};`;
        } else if (dataDtor.channels > 1) {
          const argus = vecDims < end ? `${utils.expandArgus(dimVar, vecDims, end)}, ` : '';
          return lodash.range(dataDtor.channels).map((channelIdx) => {
            return `${dataOutVar}(${utils.expandArgus(dimVar, end)}, ${channelIdx}) = ${cvVar}${utils.expandArrayAccessor(dimVar, vecDims)}${dataDtor.accessor(argus + ", " + channelIdx)};`;
          }).join('\n');
        } else {
          throw new Error(`Invalid channel number: ${dataDtor.channels}`);
        }
      }

      return `for (int ${dimVar + start} = 0; ${dimVar + start} < ${dimSizeOutVar + start}; ${dimVar + start}++) {
      ${loopVecOfMat(++start, end, vecDims, dataDtor)}
      }`;
    }

    function loopMat(start, end, dataDtor) {
      if (start === end) {
        const dtype = dataDtor.format === 'cv' ? types.cvToStd(dataDtor.dtype) : dataDtor.dtype;
        if (dataDtor.channels === 1) {
          return `${dataOutVar}(${utils.expandArgus(dimVar, end)}) = ${cvVar}${dataDtor.accessor(utils.expandArgus(dimVar, end))};`;
        } else if (dataDtor.channels > 1) {
          return lodash.range(dataDtor.channels).map((channelIdx) => {
            return `${dataOutVar}(${utils.expandArgus(dimVar, end)}, ${channelIdx}) = ${cvVar}${dataDtor.accessor(utils.expandArgus(dimVar, end))}[${channelIdx}];`;
          }).join('\n');
        } else {
          throw new Error(`Invalid channel number: ${dataDtor.channels}`);
        }
      }

      return `for (int ${dimVar + start} = 0; ${dimVar + start} < ${dimSizeOutVar + start}; ${dimVar + start}++) {
      ${loopMat(++start, end, dataDtor)}
      }`;
    }
  }
};

function renderKernelTemp(opsMeta) {
  // TODO: Validates the opsMeta format
  let parsedOpsMeta = {
    srcFile:  opsMeta.srcFile,
    opName:   opsMeta.opName,
    fnName:   opsMeta.fnName
  };
  if (opsMeta.inputs) {
    parsedOpsMeta.inputs = Object.keys(opsMeta.inputs).map((key) => {
      let inputs = opsMeta.inputs[key];
      const shape = parser.parseShape(inputs.shape);
      inputs.name = key.trim();
      inputs.typeFormat = shape.dataDtor.format;
      inputs.dtype = shape.dataDtor.dtype;
      inputs.pShape = shape;
      return inputs;
    });
  }
  if (opsMeta.outputs) {
    parsedOpsMeta.outputs = Object.keys(opsMeta.outputs).map((key) => {
      let outputs = opsMeta.outputs[key];
      const shape = parser.parseShape(outputs.shape);
      outputs.name = key.trim();
      outputs.typeFormat = shape.dataDtor.format;
      outputs.dtype = shape.dataDtor.dtype;
      outputs.pShape = shape;
      return outputs;
    });
  }
  if (opsMeta.inputoutputs) {
    parsedOpsMeta.inputoutputs = Object.keys(opsMeta.inputoutputs).map((key) => {
      let inputoutputs = opsMeta.inputoutputs[key];
      const shape = parser.parseShape(inputoutputs.shape);
      inputoutputs.name = key.trim();
      inputoutputs.typeFormat = shape.dataDtor.format;
      inputoutputs.dtype = shape.dataDtor.dtype;
      inputoutputs.pShape = shape;
      return inputoutputs;
    });
  }
  if (opsMeta.attributes) {
    parsedOpsMeta.attributes = Object.keys(opsMeta.attributes).map((key) => {
      let attributes = opsMeta.attributes[key];
      const typeExpr = utils.parseAttrType(attributes.type);
      attributes.name = key.trim();
      attributes.type = typeExpr.type;
      attributes.defaultVal = typeExpr.defaultVal;
      return attributes;
    });
  }

  const view = {
    srcFile: opsMeta.srcFile,
    opName: opsMeta.opName,
    fnName: opsMeta.fnName,
    device: opsMeta.device ? opsMeta.device : 'DEVICE_CPU',
    registerOpInput: registerOpInput(parsedOpsMeta),
    registerOpInputFn,
    registerOpOutput: registerOpOutput(parsedOpsMeta),
    registerOpOutputFn,
    opAttributes: opAttributes(parsedOpsMeta),
    registerOpAttrFn,
    getAttributesFn,
    declareAttributesFn,
    registerOpShape: registerOpShape(parsedOpsMeta),
    registerOpShapeFn,
    computeInput: computeInput(parsedOpsMeta),
    computeInputFn,
    computeExecute: computeExecute(parsedOpsMeta),
    computeExecuteFn,
    computeOutput: computeOutput(parsedOpsMeta),
    computeOutputFn
  };

  return Mustache.render(template.getKernelTemp(), view);
}

function renderpyWrapperTemp(kernelSharedLibName, opsMeta) {
  const view = {
    kernelSharedLib: kernelSharedLibName,
    ops: opsMeta.map((obj) => {
      return { name: lowerAndSnake(obj.opName) };
    })
  };

  return Mustache.render(template.getPyWrapperTemp(), view);
}

function render(url, callback) {
  try {
    const parsed = path.parse(url);

    let defaultName = parsed.name;
    let opsMeta = JSON.parse(fs.readFileSync(url));
    if (typeof opsMeta === 'object' && !Array.isArray(opsMeta)) { opsMeta = [ opsMeta ]; }
    if (!Array.isArray(opsMeta)) { return callback(new Error('Invalid JSON format')); }
    opsMeta.forEach((meta) => {
      meta.srcFile= meta.srcFile? meta.srcFile : defaultName;
      meta.opName = meta.opName ? capitalAndCamel(meta.opName) : capitalAndCamel(defaultName);
      meta.fnName = meta.fnName ? meta.fnName : defaultName;
    });

    async.parallel({
      renderKernel: (callback) => {
        async.each(opsMeta, (meta, callback) => {
          fs.writeFile(`${parsed.dir}/${lowerAndSnake(meta.opName)}_op.cc`,
                       renderKernelTemp(meta), callback);
        }, (err) => {
          if (err) { callback(err); }
          callback();
        });
      },
      renderPyWrapper: (callback) => {
        const kernelSharedLibName = `prov_${lowerAndSnake(defaultName)}_op_kernel.so`;
        fs.writeFile(`${parsed.dir}/${lowerAndSnake(defaultName)}_op.py`,
                     renderpyWrapperTemp(kernelSharedLibName, opsMeta), callback);
      }
    }, (err) => {
      if (err) { return callback(err); }
      callback();
    });
  } catch (err) {
    callback(err);
  }
}

module.exports = {
  render
};
