'use strict';
const Mustache = require('mustache');
const lodash = require('lodash');
const changeCase = require('change-case');
const utils = require('./utils');
const types = require('./types');
const template = require('./template');
const fs = require('fs');
const path = require('path');


/**
 * Define template variables.
 */
const nameVar     = '{{name}}';
const tensorInVar = '{{name}}_in';
const tensorOutVar= '{{name}}_out';
const dataInVar   = '{{name}}_in_data';
const dataOutVar  = '{{name}}_out_data';
const cvVar       = '{{name}}_cv';
const dimVar      = '{{name}}_dims_';
const dimSizeVar  = '{{name}}_dims_size_';
const shapeVar    = 'mat_shape';

/**
 * Parses shape array.
 *
 * @param {arary} shape The shape array to be parsed.
 * @return {object} The parsed shape object.
 * @example
 *
 *
 */
function parseShape(shape) {
  if (!Array.isArray(shape)) {
    throw new Error('Invalid argument: the input argument should be an array.');
  }
  if (shape.length === 0) {
    throw new Error('Invalid argument: empty shape array.');
  }

  const lastElemIdx = shape.length - 1;
  const origDataDtor = shape[lastElemIdx];
  const origDimDtorArr = shape.slice(0, lastElemIdx);

  return {
    dataDtor: utils.parseDataDtor(origDataDtor),
    dimDtorArr:  lastElemIdx === 0 ? [] : origDimDtorArr.map(utils.parseDimDtor),
    tfRank: utils.getTfRank(shape),
    cvRank: utils.getCvRank(shape)
  };
}

function ascendingId(a, b) {
  return a.id - b.id;
}

function lowerAndSnake(str) {
  return changeCase.lowerCase(changeCase.snakeCase(str));
}

const registerOpInput = function(opsMeta) {
  let result = [];
  if (opsMeta.inputs) {
    result = result.concat(opsMeta.inputs);
  }
  if (opsMeta.inputoutputs) {
    result = result.concat(opsMeta.inputoutputs.map((obj) => {
      return Object.assign({}, obj, { name: `${obj.name}_in` });
    }));
  }
  result.sort(ascendingId);
  return result.map((obj) => {
    let type = undefined;
    if (obj.typeFormat === 'cv') {
      type = types.cvToTf(obj.type);
    } else if (obj.typeFormat === 'std') {
      type = types.stdToTf(obj.type);
    }
    return { name: obj.name, type };
  });
};

const registerOpInputFn = function() {
  return `.Input("${lowerAndSnake(this.name)}: ${this.type}")`;
};

const registerOpOutput = function(opsMeta) {
  let result = [];
  if (opsMeta.outputs) {
    result = result.concat(opsMeta.outputs);
  }
  if (opsMeta.inputoutputs) {
    result = result.concat(opsMeta.inputoutputs);
  }
  result.sort(ascendingId);
  return result.map((obj) => {
    let type = undefined;
    if (obj.typeFormat === 'cv') {
      type = types.cvToTf(obj.type);
    } else if (obj.typeFormat === 'std') {
      type = types.stdToTf(obj.type);
    }
    return { name: obj.name, type };
  });
};

const registerOpOutputFn = function() {
  return `.Output("${lowerAndSnake(this.name)}: ${this.type}")`;
};

const opAttributes = function(opsMeta) {
  let result = [];
  if (opsMeta.attributes) {
    result = result.concat(opsMeta.attributes);
  }
  result.sort(ascendingId);
  return result.map((obj) => {
    return { name: obj.name, type: obj.type, defaultVal: obj.defaultVal };
  });
};

const registerOpAttrFn = function() {
  const type = this.defaultVal ? `${this.type} = ${this.defaultVal}` : `${this.type}`;
  return `.Attr("${this.name}: ${type}")`;
};

const getAttributesFn = function() {
  return `OP_REQUIRES_OK(context, context->GetAttr("${this.name}", &${this.name}_));`;
};

const declareAttributesFn = function() {
  return `${this.type}      ${this.name}_;`;
}

const registerOpShape = function(opsMeta) {
  let result = [];
  if (opsMeta.outputs) {
    result = result.concat(opsMeta.outputs);
  }
  if (opsMeta.inputoutputs) {
    result = result.concat(opsMeta.inputoutputs);
  }
  result.sort(ascendingId);
  return result.map((obj, index) => {
    return { regIdx: index, shape: obj.shape };
  });
};

const registerOpShapeFn = function() {
  const tfRank = utils.getTfRank(this.shape);
  let shapeStr;
  if (tfRank === 0) {
    shapeStr = `c->Scalar()`;
  }
  else if (tfRank === 1) {
    shapeStr = `c->Vector(${getDim(this.shape[0])})`;
  }
  else if (tfRank === 2) {
    // There two possible different shapes can have the same tfrank 2, which are:
    //  1. Two dimensional descriptors with a single channel data descriptor, eg. [ none, none, CV_64F ]
    //  2. One dimentional descriptor with a multichannel data descriptor, eg. [ none, CV_64FC3 ]
    // We need to figure out which one is the case.
    const dtor = utils.parseDataDtor(this.shape[this.shape.length - 1]);
    let secDim;
    if (dtor.channels === 1) {
      secDim = getDim(this.shape[1]);
    } else {
      secDim = dtor.channels;
    }
    shapeStr = `c->Matrix(${getDim(this.shape[0])}, ${secDim})`;
  }
  else if (tfRank >= 3) {
    const argus = this.shape.slice(0, this.shape.length - 1).map((elem) => {
      return getDim(elem);
    }).join(', ');
    shapeStr = `c->MakeShape({ ${argus} })`;
  }
  else {
    throw new Error(`Invalid rank number of the shape: ${tfRank}`);
  }
  return `c->set_output(${this.regIdx}, ${shapeStr});`;

  function getDim(dimDtor) {
    const dimStr = utils.parseDimDtor(dimDtor).dims;
    return dimStr === 'none' ? 'InferenceContext::kUnknownDim' : parseInt(dimStr);
  }
};

const computeInput = function(opsMeta) {
  let result = [];
  if (opsMeta.inputs) {
    result = result.concat(opsMeta.inputs);
  }
  if (opsMeta.inputoutputs) {
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
  {{{decDimSize}}}
  {{{cvtTensorToCv}}}
  `;
  const view = {
    name: this.name,
    regIdx: this.regIdx,
    tensorInRank: this.pShape.tfRank,
    tensorInVar: Mustache.render(tensorInVar, { name: this.name }),
    dataInVar: Mustache.render(dataInVar, { name: this.name }),
    tensorDtype: this.pShape.dataDtor.format === 'cv' ? types.cvToStd(this.pShape.dataDtor.type) :
                                                        this.pShape.dataDtor.type,
    decDimSize: Mustache.render(declareDimSize(this.pShape.tfRank), { name: this.name }),
    cvtTensorToCv: Mustache.render(convertTensorToCvByShape(this.pShape), { name: this.name })
  }
  return Mustache.render(template, view);

  function declareDimSize(rank) {
    return lodash.range(rank).map((idx) => {
      return `const int32 ${dimSizeVar}${idx} = static_cast<int32>(${tensorInVar}.dim_size(${idx}));`;
    }).join('\n');
  }

  /**
   * Generates convertion code, given the parsed CV shape object, for copying data from a tensor to a
   * CV data structure.
   *
   * Rules:
   *  - The shape array should have at least two elements, one for dimension descriptor, the other for
   *     data cell descriptor.
   *      - The format of the dimension descriptor should be: <dimension_type>:<the_length_of_dimension>.
   *        For example: "vector:3" means it's a 3-dimensional vector.
   *        - The length of dimension can be set as "none", which means the length is determined at
   *          runtime.
   *        - The type of dimension can be absence, which means it's a dimension of a Mat object.
   *      - The format of the data cell descriptor can be one of the following:
   *        1. CV_<bit-depth>{U|S|F}C(<number_of_channels>)
   *        2. primary types, eg. float, int, ...
   *  - A shape array may have multiple dimension descriptors but can have only one data cell descriptor.
   *  - The data cell descriptor should always be the last element in the shape array.
   *  - If a shape array has vector dimension(s), it/they should always start from the first element.
   *    - ie. we do not support Mat of vector.
   *  - If a shape array has multiple vector dimensions, they should be continuous.
   *    - ie. we do not support forms like vector of Mat of vector.
   *
   *  There are only four possible cases:
   *    1. Vector... of CV type object.
   *    2. Vecotr... of primary type object.
   *    3. Vector... of Mat.
   *    4. Just Mat.
   *
   * @param {array} shape The array of strings specifiying an n-dimentional CV array shape.
   * @returns {string} Returns the convertion code string.
   * @example
   *
   */
  function convertTensorToCvByShape(shape) {
    const dataDtor = shape.dataDtor;
    const dimDtorArr = shape.dimDtorArr;
    const tfRank = shape.tfRank;
    const cvRank = shape.cvRank;

    let declareStr, loopStr;
    const lastVecIdx = utils.lastIndexOf(dimDtorArr, (item) => { return item.type === 'vector'; });
    if (lastVecIdx !== -1) {
      const vecLayers = lastVecIdx + 1;
      if (vecLayers === cvRank) {
        // No Mat been declared.
        if (dataDtor.format === 'cv') {
          /*
           * Case: Vector... of vector
           *
           */
          const declareType = declareVec(vecLayers, `Vec<${types.cvToStd(dataDtor.type)}, ${dataDtor.channels}>`);
          declareStr = `${declareType} ${cvVar}(${dimSizeVar + "0"});`;
          loopStr = `${loopVecOfCv(0, cvRank, dataDtor)}`;
        } else {
          /*
           * Case: Vector... of primary type
           *
           */
          const declareType = declareVec(vecLayers, `${dataDtor.type}`);
          declareStr = `${declareType} ${cvVar}${vecLayers === 1 ? "" : (dimSizeVar + "0")};`;
          loopStr = `${loopVecOfPrim(0, cvRank)}`;
        }
      } else {
        /*
         * Case: Vector... of Mat
         *
         * // vector<Mat> a_cv(size);
         * // int mat_shape[] = { a_dims_size_1, a_dims_size_2 };
         * //
         */
        const vecRank = lastVecIdx + 1;
        const matRank = cvRank - vecRank;
        const declareType = declareVec(vecLayers, 'Mat');
        declareStr = `${declareType} ${cvVar}(${dimSizeVar + "0"});`;
        loopStr = `int ${shapeVar}[] = { ${utils.expandArgus(dimSizeVar, vecRank, cvRank)} };
        ${loopVecOfMat(0, vecRank, matRank, dataDtor)}`;
      }

      return `${declareStr}
      ${loopStr}`;
    } else {
      /*
       * Case: Just Mat
       */
      const matDtype = dataDtor.format === 'cv' ? dataDtor.toOrigStr() : types.stdToCv(dataDtor.toOrigStr());
      declareStr = `const int ${nameVar}_shape[] = { ${utils.expandArgus(dimSizeVar, cvRank)} };
      Mat ${cvVar}(${cvRank}, ${nameVar}_shape, ${matDtype});`;
      loopStr = `${loopMat(0, cvRank, dataDtor, { cvVar, dimVar, dimSizeVar, dataInVar })}`;

      return `${declareStr}
      ${loopStr}`;
    }

    function declareVec(layer, core) {
      if (layer === 0) {
        return `${core}`;
      }
      return `vector<${declareVec(--layer, core)}>`;
    }

    /**
     * Generates a nested loops for copying data from a tensor to a vector of CV object.
     *
     * @param {number} start The start index of the outermost loop.
     * @param {number} end The end index of the innermost loop.
     * @param {object} dataDtor The data descriptor that used to construct the innermost loop body.
     * @returns {string} Returns the nested loops string.
     * @example
     *
     * // input shape: [ vector:none, vector:none, CV_64FC2 ]
     * const dataDtor = { type: '64F', channels: 2 };
     * loopVecOfCv(0, 2, dataDtor);
     * // =>
     * // for (int a_dims_0 = 0; a_dims_0 < a_dims_size_0; a_dims_0++) {
     * //   for (int a_dims_1 = 0; a_dims_1 < a_dims_size_1; a_dims_1++) {
     * //     if (isnan(a_data(a_dims_0, a_dims_1, 0))) { break; }
     * //     a_cv[a_dims_0].push_back(Vec<double, 2>(
     * //       a_data(a_dims_0, a_dims_1, 0),
     * //       a_data(a_dims_0, a_dims_1, 1)
     * //     ));
     * //   }
     * // }
     */
    function loopVecOfCv(start, end, dataDtor) {
      if (start === end) {
        const l = utils.expandArgus((channelIdx) => {
          return `${dataInVar}(${utils.expandArgus(dimVar, end)}, ${channelIdx})`;
        }, dataDtor.channels);
        const data = `Vec<${types.cvToStd(dataDtor.type)}, ${dataDtor.channels}>(${l})`;
        return `if (isnan(${dataInVar}(${utils.expandArgus(dimVar, end) + ', 0'}))) { break; }
        ${cvVar}${utils.expandArrayAccessor(end - 1, dimVar)}.push_back(${data});`;
      }

      return `for (int ${dimVar + start} = 0; ${dimVar + start} < ${dimSizeVar + start}; ${dimVar + start}++) {
        ${loopVecOfCv(++start, end, dataDtor)}
      }`;
    }

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
     * // for (int a_dims_0 = 0; a_dims_0 < a_dims_size_0; a_dims_0++) {
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
     * // for (int a_dims_0 = 0; a_dims_0 < a_dims_size_0; a_dims_0++) {
     * //   a_cv.push_back(a_data(a_dims_0));
     * // }
     */
    function loopVecOfPrim(start, end) {
      if (start === end) {
        let guardStr = '';
        if (end > 1) {
          guardStr = `if (isnan(${dataInVar}(${utils.expandArgus(dimVar, end) + ', 0'}))) { break; }`;
        }
        const data = `${dataInVar}(${utils.expandArgus(dimVar, end)})`;
        return `${guardStr}
        ${cvVar}${utils.expandArrayAccessor(end - 1, dimVar)}.push_back(${data});`;
      }

      return `for (int ${dimVar + start} = 0; ${dimVar + start} < ${dimSizeVar + start}; ${dimVar + start}++) {
        ${loopVecOfPrim(++start, end)}
      }`;
    }

    /**
     * Generates a nested loops for copying data from a tensor to a vector of Mat object.
     *
     * @param {number} start The start index of the outermost loop.
     * @param {number} vecRank The rank number of the vector part in the input shape.
     * @param {number} matRank The rank number of the Mat part in the input shape.
     * @param {object} dataDtor The data descriptor that used to construct the innermost loop body.
     * @returns {string} Returns the nested loops string.
     * @example
     *
     * // input shape: [ vector:none, none, none, CV_64FC3 ]
     * //
     * // Prerequisite declaration:
     * //
     * //   vector<Mat> a_cv(size);
     * //   int mat_shape[] = { a_dims_size_1, a_dims_size_2 };
     * //
     * // |<- vecRank ->|<- matRank ->|
     * // |      1      |      2      |
     * // -----------------------------
     * // |     Sizes/dims index      |
     * // | 0           , 1   , 2     |
     * // [ vector:none , none, none  , CV_64FC3 ]
     * const vecRank = 1;
     * const matRank = 2;
     * const dataDtor = { type: '64F', channels: 3 };
     * loopVecOfMat(0, vecRank, matRank, dataDtor);
     * // =>
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
    function loopVecOfMat(start, vecRank, matRank, dataDtor) {
      if (start === vecRank) {
        let guardStr = '';
        if (vecRank > 1) {
          guardStr = `if (isnan(${dataInVar}(${utils.expandArgus(dimVar, vecRank) + ', 0'}))) { break; }`;
        }
        const data = `${dataInVar}(${utils.expandArgus(dimVar, cvRank)})`;
        const tempVar = 'mat';
        const type = dataDtor.channels > 1 ? `CV_${dataDtor.type}C${dataDtor.channels}` : `CV_${dataDtor.type}`;
        return `${guardStr}
        Mat ${tempVar}(${matRank}, ${shapeVar}, ${type});
        ${loopMat(vecRank, matRank + vecRank, dataDtor, vecRank)}
        ${cvVar}${utils.expandArrayAccessor(vecRank - 1, dimVar)}.push_back(${tempVar});`;
      }

      return `for (int ${dimVar + start} = 0; ${dimVar + start} < ${dimSizeVar + start}; ${dimVar + start}++) {
        ${loopVecOfMat(++start, vecRank, matRank, dataDtor)}
      }`;
    }

    /**
     * Generates a nested loops for copying data from a tensor to a Mat object.
     *
     * @param {number} start The start index of the outermost loop.
     * @param {number} end The end index of the innermost loop.
     * @param {object} dataDtor The data descriptor that used to construct the innermost loop body.
     * @param {number} [cvStartOffset=0] The start data index offset of the Mat object when copying the data.
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
    function loopMat(start, end, dataDtor, cvStartOffset) {
      if (start === end) {
        const cvArgus = utils.expandArgus(`${dimVar}`, cvStartOffset, end);
        const dataArgus = utils.expandArgus(`${dimVar}`, end);
        if (dataDtor.channels === 1) {
          return `${cvVar}.at<${types.cvToStd(dataDtor.type)}>(${cvArgus}) = ${dataInVar}(${dataArgus});`;
        } else if (dataDtor.channels > 1) {
          return lodash.range(dataDtor.channels).map((channelIdx) => {
            return `${cvVar}.at<Vec<${types.cvToStd(dataDtor.type)}, ${dataDtor.channels}>>(${cvArgus})[${channelIdx}] = ${dataInVar}(${dataArgus}, ${channelIdx});`;
          }).join('\n');
        } else {
          throw new Error(`Invalid channel number: ${dataDtor.channels}`);
        }
      }

      return `for (int ${dimVar + start} = 0; ${dimVar + start} < ${dimSizeVar + start}; ${dimVar + start}++) {
        ${loopMat(++start, end, dataDtor, cvStartOffset)}
      }`;
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
      return { id: obj.id, name: obj.name };
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
    declareOutputStr = 'Mat ' + outputs.map((item) => { return item.name; }).join(', ') + ';';
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
  if (opsMeta.outputs) {
    result = result.concat(opsMeta.outputs);
  }
  if (opsMeta.inputoutputs) {
    result = result.concat(opsMeta.inputoutputs);
  }
  result.sort(ascendingId);
  return result.map((obj, index) => {
    return { regIdx: index, name: obj.name, shape: obj.shape, pShape: obj.pShape };
  });
};

const computeOutputFn = function() {
  let tfShapeArgus;
  if (this.pShape.dataDtor.channels > 1) {
    tfShapeArgus = utils.expandArgus((idx) => { return `${cvVar}.size[${idx}]`; }, this.pShape.cvRank) + `, ${cvVar}.channels()`;
  } else {
    tfShapeArgus = utils.expandArgus((idx) => { return `${cvVar}.size[${idx}]`; }, this.pShape.tfRank);
  }

  const template = `
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
    tensorDtype: this.pShape.dataDtor.format === 'cv' ? types.cvToStd(this.pShape.dataDtor.type) :
                                                        this.pShape.dataDtor.type,
    cvtCvToTensor: Mustache.render(convertCvToTensorByShape(this.pShape), { name: this.name })
  };

  return Mustache.render(template, view);

  function convertCvToTensorByShape(shape) {
    const dataDtor = shape.dataDtor;
    const dimDtorArr = shape.dimDtorArr;
    const tfRank = shape.tfRank;
    const cvRank = shape.cvRank;

    return `${loop(0, cvRank, shape.dataDtor)}`;

    /**
     * Generates a nested loops for copying data from a CV Mat object to a tensor.
     *
     * @param {number} start The start index of the outermost loop.
     * @param {number} end The end index of the innermost loop.
     * @param {object} dataDtor The data descriptor that used to construct the innermost loop body.
     * @returns {string} Returns the nested loops string.
     * @example
     *
     * // input shape: [ 4, CV_64FC3 ]
     * const dataDtor = { type: 'double', channels: 3 };
     * loop(0, 1, dataDtor);
     * // =>
     * // for (int a_dims_0 = 0; a_dims_0 < a_cv.size[0]; a_dims_0++) {
     * //   a_data(a_dims_0, 0) = r_cv.at<double>(a_dims_0, 0);
     * //   a_data(a_dims_0, 1) = r_cv.at<double>(a_dims_0, 1);
     * //   a_data(a_dims_0, 2) = r_cv.at<double>(a_dims_0, 2);
     * // }
     *
     * // input shape: [ 3, 3, double ]
     * const dataDtor = { type: 'double', channels: 1 };
     * loop(0, 2, dataDtor);
     * // =>
     * // for (int a_dims_0 = 0; a_dims_0 < a_cv.size[0]; a_dims_0++) {
     * //   for (int a_dims_1 = 0; a_dims_1 < a_cv.size[1]; a_dims_1++) {
     * //     a_data(a_dims_0, a_dims_1) = r_cv.at<double>(a_dims_0, a_dims_1);
     * //   }
     * // }
     *
     * // input shape: [ 3, 3, CV_64FC2 ]
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
    function loop(start, end, dataDtor) {
      if (start === end) {
        const dtype = dataDtor.format === 'cv' ? types.cvToStd(dataDtor.type) : dataDtor.type;
        if (dataDtor.channels === 1) {
          return `${dataOutVar}(${utils.expandArgus(dimVar, end)}) = ${cvVar}.at<${dtype}>(${utils.expandArgus(dimVar, end)});`;
        } else if (dataDtor.channels > 1) {
          return lodash.range(dataDtor.channels).map((channelIdx) => {
            return `${dataOutVar}(${utils.expandArgus(dimVar, end)}, ${channelIdx}) = ${cvVar}.at<${dtype}>(${utils.expandArgus(dimVar, end)}, ${channelIdx});`;
          }).join('\n');
        } else {
          throw new Error(`Invalid channel number: ${dataDtor.channels}`);
        }
      }

      return `for (int ${dimVar + start} = 0; ${dimVar + start} < ${cvVar}.size[${start}]; ${dimVar + start}++) {
      ${loop(++start, end, dataDtor)}
      }`;
    }
  }
};

function _render(opsMeta) {
  // TODO: Validates the opsMeta format
  let parsedOpsMeta = {
    opName: opsMeta.opName,
    fnName: opsMeta.fnName
  };
  if (opsMeta.inputs) {
    parsedOpsMeta.inputs = Object.keys(opsMeta.inputs).map((key) => {
      let inputs = opsMeta.inputs[key];
      const shape = parseShape(inputs.shape);
      inputs.name = key.trim();
      inputs.typeFormat = shape.dataDtor.format;
      inputs.type = shape.dataDtor.type;
      inputs.pShape = shape;
      return inputs;
    });
  }
  if (opsMeta.outputs) {
    parsedOpsMeta.outputs = Object.keys(opsMeta.outputs).map((key) => {
      let outputs = opsMeta.outputs[key];
      const shape = parseShape(outputs.shape);
      outputs.name = key.trim();
      outputs.typeFormat = shape.dataDtor.format;
      outputs.type = shape.dataDtor.type;
      outputs.pShape = shape;
      return outputs;
    });
  }
  if (opsMeta.inputoutputs) {
    parsedOpsMeta.inputoutputs = Object.keys(opsMeta.inputoutputs).map((key) => {
      let inputoutputs = opsMeta.inputoutputs[key];
      const shape = parseShape(inputoutputs.shape);
      inputoutputs.name = key.trim();
      inputoutputs.typeFormat = shape.dataDtor.format;
      inputoutputs.type = shape.dataDtor.type;
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

  return Mustache.render(template.get(), view);
}

function render(url, callback) {
  try {
    const parsed = path.parse(url);

    let opsMeta;
    opsMeta = JSON.parse(fs.readFileSync(url));
    opsMeta.opName = opsMeta.opName ? changeCase.upperCaseFirst(changeCase.camelCase(opsMeta.opName)) :
                                      changeCase.upperCaseFirst(changeCase.camelCase(parsed.name));
    opsMeta.fnName = opsMeta.fnName ? changeCase.snakeCase(opsMeta.fnName) :
                                      changeCase.snakeCase(parsed.name);

    const output = _render(opsMeta);
    fs.writeFile(`${parsed.dir}/${parsed.name}_op.cc`, output, callback);
  } catch (err) {
    callback(err);
  }
}

module.exports = {
  render
};
