'use strict';
const lodash = require('lodash');
const utils = require('./utils');
const types = require('./types');

/**
 * Define constant variables.
 */
const SCALAR      = 'SCALAR';
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
 *   1. Scalar... of primary type object.
 *   2. Vecotr... of primary type object.
 *   3. Vector... of Mat object.
 *   4. Just Mat object.
 *
 * We are going to parse the following information from the shape array:
 *
 *  * type: The type of one of the above cases.
 *  * varDecStr: The declare string of the type.
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
    dataDtor: parseDataDtor(origDataDtor),
    dimDtorArr:  lastElemIdx === 0 ? [] : origDimDtorArr.map(parseDimDtor),
    tfRank: getTfRank(shape),
    cvRank: getCvRank(shape)
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
  result.matDimDtorArr = result.dimDtorArr.slice(lastVecIdx + 1, result.dimDtorArr.length);

  if (lastVecIdx !== -1) {
    if (result.vecDims === result.cvRank) {
      if (result.dataDtor.format === 'cv') {
        // Vectors of primary does not support using CV string as data descriptor.
        throw new Error('Invalid shape format: vector of OpenCV type is not allowed');
      }
      result.type = VEC_OF_PRIM;
    } else {
      if (result.dataDtor.format !== 'cv') {
        // Vectors of Mat(x) requires to use CV string as data descriptor.
        throw new Error('Invalid shape format: Mat of primary type is not allowed');
      }
      result.type = VEC_OF_MAT;
    }
  } else {
    if (result.dataDtor.format === 'cv') {
      if (result.cvRank === 0) {
        throw new Error('Invalid shape format: scalar of OpenCV type is not allowed');
      }
      result.type = MAT;
    } else {
      if (result.cvRank > 0) {
        throw new Error('Invalid shape format: Mat of primary type is not allowed');
      }
      result.type = SCALAR;
    }
  }

  switch (result.type) {
      case VEC_OF_PRIM:
          result.varDecStr = declareVec(`${result.dataDtor.dtype}`, result.vecDims);
          break;
      case VEC_OF_MAT:
      case MAT:
          const ctype = result.dataDtor.ctype;
          if (ctype === 'Matx' || ctype === 'Vec') {
            // It's static type Mat, we should parse the structure/dimensions of the shape.
            if (result.matDimDtorArr.find((elem) => { return elem.dims === 'none'; })) {
              throw new Error('Invalid shape format: dynamic dimension size is not allowed for static Mat type');
            }
            if (result.matDims > 1 && ctype === 'Vec') {
              throw new Error('Invalid shape format: Vec type should be one-dimensional');
            }

            let matDimStr = result.matDimDtorArr.map((elem) => { return elem.dims; }).join(', ');
            if (ctype === 'Matx' && result.matDims === 1) {
              // It's a vector so we have to add a '1' at the front.
              matDimStr = `1, ${matDimStr}`;
            }
            result.varMatDecStr = `${ctype}<${types.cvToStd(result.dataDtor.dtype)}, ${matDimStr}>`;
          } else {
            result.varMatDecStr = 'Mat';
          }

          if (result.type === VEC_OF_MAT) {
            result.varDecStr = declareVec(`${result.varMatDecStr}`, result.vecDims);
          } else if (result.type === MAT) {
            result.varDecStr = result.varMatDecStr;
          }
          break;
      case SCALAR:
          result.varDecStr = result.dataDtor.dtype;
          break;
      default:
          throw new Error('Unknown shape format: ' + shape);
  }

  return result;

  function declareVec(template, end) {
    if (end === 0) {
      return `${template}`;
    }
    return `vector<${declareVec(template, --end)}>`;
  }
}

/**
 * Parses dimensional descriptor string.
 *
 * The format of the shape element can be one of the following:
 *  1. <type>:<dimensions>
 *  2. <dimensions>
 *
 * @param {array} shape The shape array to be parsed.
 * @returns {object} The parsed object.
 */
function parseDimDtor(str) {
  if (!lodash.isString(str) && !lodash.isInteger(str)) {
    throw new Error(`Invalid dimensional descriptor format: ${str}`);
  }
  const format1 = /(^vector):(none|0|[1-9]\d*)$/;
  const format2 = /(^none|^0|^[1-9]\d*)$/;

  let result;
  if (format1.test(str)) {
    result = format1.exec(str);
    if (result[2] !== 'none' && parseInt(result[2]) < 1) {
      throw new Error('Invalid dimensional descriptor format: dimension should > 0');
    }
    return {
      type: result[1],
      dims: result[2] === 'none' ? result[2] : parseInt(result[2])
    };
  } else if (format2.test(str)) {
    result = format2.exec(str);
    if (result[1] !== 'none' && parseInt(result[1]) < 1) {
      throw new Error('Invalid dimensional descriptor format: dimension should > 0');
    }
    return {
      type: 'Mat',
      dims: result[1] === 'none' ? result[1] : parseInt(result[1])
    };
  } else {
    throw new Error(`Invalid dimensional descriptor format: ${str}`);
  }
}

/**
 * Parses data descriptor string.
 *
 * Data descriptor format can be one of the following:
 *  1. CV_<bit-depth>{U|S|F}C(<number_of_channels>):{Mat|Matx|Vec}
 *  2. CV_<bit-depth>{U|S|F}:{Mat|Matx|Vec}  // which means channels = 1
 *  3. Primary types, eg. int, float.
 *
 * @param {string} str The data descriptor string to be parsed.
 * @return {object} The parsed object.
 * @example
 *
 * parseDataDtor('CV_32FC2');
 * // => { format: 'cv', ctype: 'Mat', dtype: '32F', channels: 2 }
 *
 * parseDataDtor('CV_32FC2:Matx');
 * // => { format: 'cv', ctype: 'Matx', dtype: '32F', channels: 2 }
 *
 * parseDataDtor('CV_64F:Vec');
 * // => { format: 'cv', ctype: 'Vec', dtype: '64F', channels: 1 }
 *
 * parseDataDtor('int');
 * // => { format: 'std', type: 'int', channels: 1 }
 *
 */
function parseDataDtor(str) {
  if (typeof str !== 'string') {
    throw new Error(`Invalid data descriptor format: ${str}`);
  }

  let parsed = { toString: function() { return this.dataDtorStr; }.bind({ dataDtorStr: str }) };

  if (str.startsWith('CV')) {
    parsed.format = 'cv';

    const dtypeFormat1 = /^CV_([1-9]\d*)(U|S|F)C([1-9]\d*)$/;
    const dtypeFormat2 = /^CV_([1-9]\d*)(U|S|F)$/;
    const ctypeFormat  = /^(Mat|Matx|Vec)$/;

    const part1 = str.split(':')[0];
    const part2 = str.split(':')[1];

    let dtype;
    if (dtypeFormat1.test(part1)) {
      dtype = dtypeFormat1.exec(part1);

      const depth = `${dtype[1]}${dtype[2]}`;
      const channels = dtype[3];
      if (channels > 4) {
        // Maximum channel number is 4
        throw new Error(`Invalid Data Cell format of channel: expect number between 1, 4 but get ${channels} from ${str}`);
      }
      if (depth !== '8U' && depth !== '8S' && depth !== '16U' && depth !== '16S'
          && depth !== '32S' && depth !== '32F' && depth !== '64F') {
        throw new Error(`Invalid Data Cell format of depth: ${depth} from ${str}`);
      }
      parsed.dtype = depth;
      parsed.channels = parseInt(channels);
    } else if (dtypeFormat2.test(part1)) {
      dtype = dtypeFormat2.exec(part1);

      const depth = `${dtype[1]}${dtype[2]}`;
      if (depth !== '8U' && depth !== '8S' && depth !== '16U' && depth !== '16S'
          && depth !== '32S' && depth !== '32F' && depth !== '64F') {
        throw new Error(`Invalid Data Cell format of depth: ${depth} from ${str}`);
      }
      parsed.dtype = depth;
      parsed.channels = 1;
    } else {
      throw new Error(`Invalid data descriptor format: ${str}`);
    }

    // Return original CV string part.
    parsed.toString = toString.bind({ dataDtorStr: part1 });

    parsed.ctype = 'Mat';   // default ctype
    if (part2) {
      if (!ctypeFormat.test(part2)) {
        throw new Error(`Invalid data descriptor format: ${str}`);
      }
      parsed.ctype = ctypeFormat.exec(part2)[1];
    }
    if (parsed.ctype !== 'Mat' && parsed.channels > 1) {
      throw new Error(`Invalid data descriptor format: static Mat type does not support multichannels`);
    }

    if (parsed.ctype === 'Matx') {
      parsed.accessor = function(argusStr) { return `(${argusStr})`; };
    } else if (parsed.ctype === 'Vec') {
      parsed.accessor = function(argusStr) { return `[${argusStr}]`; };
    } else {
      // Default is Mat.
      parsed.accessor = function(argusStr) {
        return `.at<${this.dtype}>(${argusStr})`;
      }.bind({ dtype: types.cvToStd(parsed.dtype) });
    }
  } else {
    const stdFormat = /^(char|int|float|double)$/;
    if (!stdFormat.test(str)) {
      throw new Error(`Invalid data descriptor format: ${str}`);
    }
    parsed.format = 'std';
    parsed.dtype = stdFormat.exec(str)[1];
    parsed.channels = 1;
    parsed.toString = toString.bind({ dataDtorStr: parsed.dtype });
  }

  return parsed;

  function toString() {
    return this.dataDtorStr;
  }
}

/**
 * Calculates the rank number of the shape in tensorflow aspect.
 *
 * @param {array} shape The shape array to be calculated.
 * @returns {number} The rank number.
 * @example
 *
 * getTfRank([ 3, 3, CV_32F ]);
 * // => 2
 *
 * getTfRank([ 3, 3, CV_32FC2 ]);
 * // => 3
 *
 * getTfRank([ 3, 3, int ]);
 * // => 2
 *
 * getTfRank([ int ]);
 * // => 0
 *
 * getTfRank([ ]);
 * // => Invalid shape format: empty shape array.
 */
function getTfRank(shape) {
  if (!shape || !Array.isArray(shape)) { throw new Error('Invalid arguments: no shape array found.'); }
  if (shape.length === 0) { throw new Error('Invalid shape format: empty shape array.'); }

  const dtor = parseDataDtor(shape[shape.length - 1]);
  const cvRank = shape.length - 1;
  if (dtor.format === 'cv') {
    if (dtor.channels === 1) {
      return cvRank;
    }
    // It's multichannel and should be count as one dimension.
    return cvRank + 1;
  } else {
    return cvRank;
  }
}

/**
 * Calculates the rank number of the shape in OpenCV aspect.
 *
 * @param {array} shape The shape array to be calculated.
 * @returns {number} The rank number.
 */
function getCvRank(shape) {
  if (!shape || !Array.isArray(shape)) { throw new Error('Invalid arguments: no shape array found.'); }
  if (shape.length === 0) { throw new Error('Invalid shape format: empty shape array.'); }

  return shape.length - 1;
}

module.exports = {
  SCALAR,
  VEC_OF_PRIM,
  VEC_OF_MAT,
  MAT,
  parseShape,
  getTfRank,
  getCvRank,
  parseDataDtor,
  parseDimDtor
};
