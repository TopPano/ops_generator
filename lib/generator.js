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

/**
 * Define constant variables.
 */
const VEC_OF_CV  = 'VEC_OF_CV';
const VEC_OF_PRIM = 'VEC_OF_PRIM';
const VEC_OF_MAT  = 'VEC_OF_MAT';
const MAT         = 'MAT';

/**
 * Parses shape array.
 *
 * Rules:
 *  - The shape array is constructed by two kinds of elements: the dimension descriptor and the
 *    data descriptor.
 *
 *    Shape array foramt:
 *
 *    [ Dim. Descriptor, Dim. Descriptor, ..., Data Descriptor ]
 *
 *    A shape array may have multiple dimension descriptors but it can only have one data descriptor.
 *    The data descriptor should always be the last element in the shape array.
 *
 *  - The format of the dimension descriptor is as following:
 *
 *    <dimension type>:<the size of the dimension>
 *
 *    Currently we only support dimension type of "vector", and the default type is "Mat" if the
 *    type of dimension is not present. In addition, the size of the dimension can be set as
 *    "none" which means the size will be determined at runtime.
 *
 *    For example: "vector:3" means it's a 3-dimensional vector.
 *
 *    If a shape array has vector dimensions, those dimensions should always start from the first
 *    element and be continued. In the other words, we don't support forms like "Mat of vector" or
 *    "vector of Mat of vector".
 *
 *  - The format of the data descriptor can be one of the following:
 *
 *    1. CV_<bit-depth>{U|S|F}C(<number_of_channels>)
 *    2. primary types, eg. float, int, ...
 *
 * There are only four possible cases for the type of a shape array:
 *   1. Vector... of CV type object.
 *   2. Vecotr... of primary type object.
 *   3. Vector... of Mat object.
 *   4. Just Mat object.
 *
 * We are going to parse the following information from the shape array:
 *
 *  * type: The type of one of the above cases.
 *  * typeDecStr: The declare string of the type.
 *  * tfRank: The tensor rank of the array.
 *  * cvRank: The CV rank of the array.
 *    - NOTE: The base of tensor rank and CV rank are not the same. Tensor rank will consider the
 *            channel number of the data descriptor, however, the CV rank won't. Though, in some
 *            cases (when channels = 1), tensor rank number will equal the CV rank number.
 *  * vecDims: The number of dimensions of the vector dimensions.
 *  * matDims: The number of dimensions of the Mat dimensions.
 *
 * For examples:
 *
 * |       Vector of Mat       |
 * -----------------------------
 * |     Sizes/dims index      |
 * | 0           , 1   , 2     |
 * -----------------------------
 * |<- vecDims ->|<- matDims ->|
 * |      1      |      2      |
 * -----------------------------
 * |<-        cvRank         ->|
 * |             3             |
 * ----------------------------------------
 * |<-              tfRank              ->|
 * |                   4                  |
 * [ vector:none , none, none  , CV_64FC3 ]
 *
 *
 * |            Mat            |
 * -----------------------------
 * |     Sizes/dims index      |
 * | 0           , 1   , 2     |
 * -----------------------------
 * |<-        matDims        ->|
 * |             3             |
 * -----------------------------
 * |<-        cvRank         ->|
 * |             3             |
 * -----------------------------
 * |<-        tfRank         ->|
 * |             3             |
 * [        none , none, none  , CV_64F ]
 *
 * @param {arary} shape The shape array to be parsed.
 * @return {object} The parsed shape object.
 *
 */
function parseShape(shape) {
  if (!Array.isArray(shape)) {
    throw new Error('Invalid arguments: no shape array found.');
  }
  if (shape.length === 0) {
    throw new Error('Invalid shape format: empty shape array.');
  }

  const lastElemIdx = shape.length - 1;
  const origDataDtor = shape[lastElemIdx];
  const origDimDtorArr = shape.slice(0, lastElemIdx);

  let result = {
    dataDtor: utils.parseDataDtor(origDataDtor),
    dimDtorArr:  lastElemIdx === 0 ? [] : origDimDtorArr.map(utils.parseDimDtor),
    tfRank: utils.getTfRank(shape),
    cvRank: utils.getCvRank(shape)
  };

  const lastVecIdx = utils.lastIndexOf(result.dimDtorArr, (item) => {
    return item.type === 'vector';
  });
  if (lastVecIdx > -1) {
    result.dimDtorArr.slice(0, lastVecIdx).forEach((item) => {
      if (item.type !== 'vector') {
        throw new Error('Invalid shape format: Mat of vector or vector of Mat of vector is not allowed');
      }
    });
  }
  result.vecDims = lastVecIdx + 1;
  result.matDims = result.cvRank - result.vecDims;
  if (lastVecIdx !== -1) {
    if (result.vecDims === result.cvRank) {
      if (result.dataDtor.format === 'cv' && result.dataDtor.channels > 1) {
        result.type = VEC_OF_CV;
        result.typeDecStr = declareVec(
          `Vec<${types.cvToStd(result.dataDtor.type)}, ${result.dataDtor.channels}>`,
          result.vecDims
        );
      } else {
        result.type = VEC_OF_PRIM;
        const dtype = result.dataDtor.format === 'cv' ? types.cvToStd(result.dataDtor.type) :
                                                        result.dataDtor.type;
        result.typeDecStr = declareVec(`${dtype}`, result.vecDims);
      }
    } else {
      result.type = VEC_OF_MAT;
      result.typeDecStr = declareVec('Mat', result.vecDims);
    }
  } else {
    result.type = MAT;
    result.typeDecStr = 'Mat';
  }

  return result;

  function declareVec(template, end) {
    if (end === 0) {
      return `${template}`;
    }
    return `vector<${declareVec(template, --end)}>`;
  }
}

function ascendingId(a, b) {
  return a.id - b.id;
}

function lowerAndSnake(str) {
  return changeCase.lowerCase(changeCase.snakeCase(str));
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
    // There two possible different shapes can have the same tfrank (>=2), which are:
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
    const dtor = utils.parseDataDtor(this.shape[this.shape.length - 1]);
    const argus = this.shape.slice(0, this.shape.length - 1).map((elem) => {
      return getDim(elem);
    }).join(', ');
    if (dtor.channels === 1) {
      shapeStr = `c->MakeShape({ ${argus} })`;
    } else {
      shapeStr = `c->MakeShape({ ${argus}, ${dtor.channels} })`;
    }
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
    tensorDtype: this.pShape.dataDtor.format === 'cv' ? types.cvToStd(this.pShape.dataDtor.type) :
                                                        this.pShape.dataDtor.type,
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

    // XXX: DO NOT initialize the vector when it is one-dimenisonal, otherwise
    //      the new value will be attached (by push_back()) after the initial values
    //      instead of replace them. However, we need to initialize the outermost vector
    //      if it's multi-dimensional.
    const vecInitStr = pShape.vecDims > 1 ? `(${dimSizeInVar + "0"})` : '';

    let declareStr, loopStr;
    switch (pShape.type) {
      case VEC_OF_CV:
        declareStr = `${pShape.typeDecStr} ${cvVar}${vecInitStr};`;
        loopStr = `${loopVecOfCv(0, cvRank, dataDtor)}`;
        break;
      case VEC_OF_PRIM:
        declareStr = `${pShape.typeDecStr} ${cvVar}${vecInitStr};`;
        loopStr = `${loopVecOfPrim(0, cvRank)}`;
        break;
      case VEC_OF_MAT:
        declareStr = `${pShape.typeDecStr} ${cvVar}${vecInitStr};`;
        loopStr = `int ${cvShapeVar}[] = { ${utils.expandArgus(dimSizeInVar, pShape.vecDims, cvRank)} };
        ${loopVecOfMat(0, pShape.vecDims, pShape.matDims, dataDtor)}`;
        break;
      case MAT:
        const matDtype = dataDtor.format === 'cv' ? dataDtor.toOrigStr() : types.stdToCv(dataDtor.toOrigStr());
        declareStr = `const int ${nameVar}_shape[] = { ${utils.expandArgus(dimSizeInVar, cvRank)} };
        ${pShape.typeDecStr} ${cvVar}(${cvRank}, ${nameVar}_shape, ${matDtype});`;
        loopStr = `${loopMat(0, cvRank, dataDtor)}`;
        break;
    }

    return `${declareStr}
    ${loopStr}`;

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
        ${cvVar}${utils.expandArrayAccessor(dimVar, end - 1)}.push_back(${data});`;
      }

      return `for (int ${dimVar + start} = 0; ${dimVar + start} < ${dimSizeInVar + start}; ${dimVar + start}++) {
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
        ${cvVar}${utils.expandArrayAccessor(dimVar, end - 1)}.push_back(${data});`;
      }

      return `for (int ${dimVar + start} = 0; ${dimVar + start} < ${dimSizeInVar + start}; ${dimVar + start}++) {
        ${loopVecOfPrim(++start, end)}
      }`;
    }

    /**
     * Generates a nested loops for copying data from a tensor to a vector of Mat object.
     *
     * @param {number} start The start index of the outermost loop.
     * @param {number} vecDims The number of dimensions of the vector part in the input shape.
     * @param {number} matDims The number of dimensions of the Mat part in the input shape.
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
    function loopVecOfMat(start, vecDims, matDims, dataDtor) {
      if (start === vecDims) {
        let guardStr = '';
        if (vecDims > 1) {
          guardStr = `if (isnan(${dataInVar}(${utils.expandArgus(dimVar, vecDims) + ', 0'}))) { break; }`;
        }
        const data = `${dataInVar}(${utils.expandArgus(dimVar, cvRank)})`;
        const tempVar = 'mat';
        const type = dataDtor.channels > 1 ? `CV_${dataDtor.type}C${dataDtor.channels}` : `CV_${dataDtor.type}`;
        return `${guardStr}
        Mat ${tempVar}(${matDims}, ${cvShapeVar}, ${type});
        ${loopMat(vecDims, matDims + vecDims, dataDtor, vecDims, tempVar)}
        ${cvVar}${utils.expandArrayAccessor(dimVar, vecDims - 1)}.push_back(${tempVar});`;
      }

      return `for (int ${dimVar + start} = 0; ${dimVar + start} < ${dimSizeInVar + start}; ${dimVar + start}++) {
        ${loopVecOfMat(++start, vecDims, matDims, dataDtor)}
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
          return `${newCvVar ? newCvVar : cvVar}.at<${types.cvToStd(dataDtor.type)}>(${cvArgus}) = ${dataInVar}(${dataArgus});`;
        } else if (dataDtor.channels > 1) {
          return lodash.range(dataDtor.channels).map((channelIdx) => {
            return `${newCvVar ? newCvVar : cvVar}.at<Vec<${types.cvToStd(dataDtor.type)}, ${dataDtor.channels}>>(${cvArgus})[${channelIdx}] = ${dataInVar}(${dataArgus}, ${channelIdx});`;
          }).join('\n');
        } else {
          throw new Error(`Invalid channel number: ${dataDtor.channels}`);
        }
      }

      return `for (int ${dimVar + start} = 0; ${dimVar + start} < ${dimSizeInVar + start}; ${dimVar + start}++) {
        ${loopMat(++start, end, dataDtor, cvStartOffset, newCvVar)}
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
    if (this.outputs.length > 0) {
      declareOutputStr = 'Mat ' + outputs.map((item) => { return item.name; }).join(', ') + ';';
    }
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
    tensorDtype: this.pShape.dataDtor.format === 'cv' ? types.cvToStd(this.pShape.dataDtor.type) :
                                                        this.pShape.dataDtor.type,
    decCvDimSize: Mustache.render(declareCvDimSize(this.pShape), { name: this.name }),
    cvtCvToTensor: Mustache.render(convertCvToTensorByShape(this.pShape), { name: this.name })
  };

  return Mustache.render(template, view);

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
   * XXX: Note: If the shape type is vector of something, we treat the all the vectors as fixed
   *            length, so we only check the size of the first element for each vector layers.
   */
  function declareCvDimSize(pShape) {
    const vecDims = pShape.vecDims;
    const matDims = pShape.matDims;
    let outStr = '', idx = 0, i;

    for (i = 0; i < vecDims; i++) {
      outStr += `const int32 ${dimSizeOutVar + idx} = static_cast<int32>(${cvVar}${expand(i)}.size());\n`;
      idx++;
    }

    for (i = 0; i < matDims; i++) {
      outStr += `const int32 ${dimSizeOutVar + idx} = static_cast<int32>(${cvVar}${expand(vecDims)}.size[${i}]);\n`;
      idx++;
    }

    if (pShape.dataDtor.channels > 1) {
      if (pShape.type === 'VEC_OF_CV') {
        outStr += `const int32 ${dimSizeOutVar + idx} = static_cast<int32>(${cvVar}${expand(vecDims)}.channels);\n`;
      } else {
        // VEC_OF_MAT or MAT
        outStr += `const int32 ${dimSizeOutVar + idx} = static_cast<int32>(${cvVar}${expand(vecDims)}.channels());\n`;
      }
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
      case VEC_OF_CV:
        loopStr = `${loopVecOfCv(0, cvRank, dataDtor)}`;
        break;
      case VEC_OF_PRIM:
        loopStr = `${loopVecOfPrim(0, cvRank)}`;
        break;
      case VEC_OF_MAT:
        loopStr = `${loopVecOfMat(0, pShape.cvRank, pShape.vecDims, dataDtor)}`;
        break;
      case MAT:
        loopStr = `${loopMat(0, cvRank, dataDtor)}`;
        break;
    }

    return `${loopStr}`;
    //return `${loop(0, pShape.cvRank, pShape.vecDims, pShape.dataDtor)}`;

    function loopVecOfCv(start, end, dataDtor) {
      if (start === end) {
        return lodash.range(dataDtor.channels).map((channelIdx) => {
          return `${dataOutVar}(${utils.expandArgus(dimVar, end)}, ${channelIdx}) = ${cvVar}${utils.expandArrayAccessor(dimVar, end)}[${channelIdx}];`;
        }).join('\n');
      }

      return `for (int ${dimVar + start} = 0; ${dimVar + start} < ${dimSizeOutVar + start}; ${dimVar + start}++) {
      ${loopVecOfCv(++start, end, dataDtor)}
      }`;
    }

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
        const dtype = dataDtor.format === 'cv' ? types.cvToStd(dataDtor.type) : dataDtor.type;
        if (dataDtor.channels === 1) {
          return `${dataOutVar}(${utils.expandArgus(dimVar, end)}) = ${cvVar}${utils.expandArrayAccessor(dimVar, vecDims)}.at<${dtype}>(${utils.expandArgus(dimVar, vecDims, end)});`;
        } else if (dataDtor.channels > 1) {
          const argus = vecDims < end ? `${utils.expandArgus(dimVar, vecDims, end)}, ` : '';
          return lodash.range(dataDtor.channels).map((channelIdx) => {
            return `${dataOutVar}(${utils.expandArgus(dimVar, end)}, ${channelIdx}) = ${cvVar}${utils.expandArrayAccessor(dimVar, vecDims)}.at<${dtype}>(${argus + channelIdx});`;
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

      return `for (int ${dimVar + start} = 0; ${dimVar + start} < ${dimSizeOutVar + start}; ${dimVar + start}++) {
      ${loopMat(++start, end, dataDtor)}
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
