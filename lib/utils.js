'use strict';
const lodash = require('lodash');


/**
 * Calculates the rank number of the shape in tensorflow aspect.
 *
 * @param {array} shpae The shape array to be calculated.
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
 *
 * getTfRank([ CV_32C2 ]);
 * // => Invalid shape format: rank 0 with multichannel is not allowed.
 *
 */
function getTfRank(shape) {
  if (!shape || !Array.isArray(shape)) { throw new Error('Invalid arguments: no shape array found.'); }
  if (shape.length === 0) { throw new Error('Invalid shape format: empty shape array.'); }

  const dtor = parseDataDtor(shape[shape.length - 1]);
  if (shape.length === 1) {
    if (dtor.format === 'cv' && dtor.channels > 1) {
      // Rank 0 with multichannel is not support.
      // [ CV_32F ]   => valid
      // [ CV_32FC2 ] => invalid
      throw new Error('Invalid shape format: rank 0 with multichannel is not allowed.');
    }
    return 0; // scalar
  }

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
 * @param {array} shpae The shape array to be calculated.
 * @returns {number} The rank number.
 */
function getCvRank(shape) {
  if (!shape || !Array.isArray(shape)) { throw new Error('Invalid arguments: no shape array found.'); }
  if (shape.length === 0) { throw new Error('Invalid shape format: empty shape array.'); }

  const dtor = parseDataDtor(shape[shape.length - 1]);
  if (shape.length === 1) {
    if (dtor.format === 'cv' && dtor.channels > 1) {
      // Rank 0 with multichannel is not support.
      // [ CV_32F ]   => valid
      // [ CV_32FC2 ] => invalid
      throw new Error('Invalid shape format: rank 0 with multichannel is not allowed.');
    }
    return 0; // scalar
  } else {
    return shape.length - 1;
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
  if (typeof str !== 'string' && typeof str !== 'number') {
    throw new Error(`Invalid dimensional descriptor format: ${str}`);
  }
  const format1 = /(vector):(none|\d)/;
  const format2 = /(none|\d)/;

  let result;
  if (format1.test(str)) {
    result = format1.exec(str);
    if (result[2] < 1) {
      throw new Error('Invalid dimensional descriptor format: dimension should > 0');
    }
    return {
      type: result[1],
      dims: result[2]
    };
  } else if (format2.test(str)) {
    result = format2.exec(str);
    if (result[1] < 1) {
      throw new Error('Invalid dimensional descriptor format: dimension should > 0');
    }
    return {
      type: 'mat',
      dims: result[1]
    };
  } else {
    throw new Error(`Invalid dimensional descriptor format: ${str}`);
  }
}

/**
 * Parses data descriptor string.
 *
 * Data descriptor format can be one of the following:
 *  1. CV_<bit-depth>{U|S|F}C(<number_of_channels>)
 *  2. CV_<bit-depth>{U|S|F}  // which means channels = 1
 *  3. Primary types, eg. int, float.
 *
 * @param {string} str The data descriptor string to be parsed.
 * @return {object} The parsed object.
 * @example
 *
 * parseDataDtor('CV_32FC2');
 * // => { format: 'cv', type: '32F', channels: 2 }
 *
 * parseDataDtor('int');
 * // => { format: 'std', type: 'int', channels: 1 }
 *
 */
function parseDataDtor(str) {
  if (typeof str !== 'string') {
    throw new Error(`Invalid data descriptor format: ${str}`);
  }
  const format1 = /CV_(8|16|32|64)(U|S|F)C(\d{1,3})/;
  const format2 = /CV_(8|16|32|64)(U|S|F)/;
  const format3 = /(char|int|float|double)/;
  let result;
  if (format1.test(str)) {
    result = format1.exec(str);
    if (result[3] < 1 || result[3] > 512) {
      // Maximum channel number is 512
      throw new Error(`Invalid Data Cell format of channel: expect number between 1, 512 but get ${result[3]}`);
    }
    if (result[2] === 'F' && (result[1] === '8' || result[1] === '16')) {
      // Not support float type under 32 bits.
      throw new Error(`Invalid Data Cell format of depth: ${result[1] + result[2]}`);
    }
    return {
      format:   'cv',
      type:     result[1] + result[2],
      channels: parseInt(result[3])
    };
  } else if (format2.test(str)) {
    result = format2.exec(str);
    if (result[2] === 'F' && (result[1] === '8' || result[1] === '16')) {
      // Not support float type under 32 bits.
      throw new Error(`Invalid Data Cell format of depth: ${result[1] + result[2]}`);
    }
    return {
      format:   'cv',
      type:     result[1] + result[2],
      channels: 1
    };
  } else if (format3.test(str)) {
    result = format3.exec(str);
    return {
      format:   'std',
      type:     result[1],
      channels: 1
    };
  } else {
    throw new Error(`Invalid data descriptor format: ${str}`);
  }
}

/**
 * Parses the attribute type expression string.
 *
 * The format of attribute type expression please see:
 * https://www.tensorflow.org/versions/r0.12/how_tos/adding_an_op/index.html#attr-types
 *
 * @param {string} str The attribute type expression string to be parsed.
 * @return {object} The parsed object.
 *
 */
function parseAttrType(str) {
  if (typeof str !== 'string') {
    throw new Error(`Invalid attribute type format: ${str}`);
  }
  // TODO: Validate the format/value of type and default value.
  const parsed = str.split('=');
  return {
    type: parsed[0].trim(),
    defaultVal: parsed[1] ? parsed[1].trim() : undefined
  };
}

function lastIndexOf(arr, callback) {
  let lastIndex = -1;
  if (!Array.isArray(arr) || !callback || typeof callback !== 'function') {
    return lastIndex;
  }
  arr.forEach((elem) => {
    if (callback(elem)) {
      lastIndex++;
    }
  });
  return lastIndex;
}

/*
 * This function will generate a list of square brackets with custom index string.
 * For example, expandArgus(3, 'var_') will return: '[var_0][var_1][var_2]'
 */
function expandArrayAccessor(length, template) {
  return lodash.range(length).map((idx) => {
    return `[${template}${idx}]`;
  }).join('');
}

/*
 * This function will generate a list of arguments with the index number as the postfix.
 * For example:
 *  expandArgus(3, 'var_')    => 'var_0, var_1, var_2'
 *  expandArgus(1, 3, 'var_') => 'var_1, var_2'
 */
function expandArgus(template, start, end) {
  if (template && typeof template === 'number') {
    throw new Error(`[expandArgus] Invalid parameters: ${template}`);
  }
  return lodash.range(start, end).map((idx) => {
    if (typeof template === 'function') {
      return template(`${idx}`);
    }
    return `${template}${idx}`;
  }).join(', ');
}

module.exports = {
  getTfRank,
  getCvRank,
  parseDimDtor,
  parseDataDtor,
  parseAttrType,
  lastIndexOf,
  expandArrayAccessor,
  expandArgus
};

