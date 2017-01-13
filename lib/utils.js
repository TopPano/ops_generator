'use strict';
const lodash = require('lodash');


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

  shape.slice(0, shape.length - 1).forEach((dimension) => {
    // Validate the dimension descriptor format
    parseDimDtor(dimension);
  });

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
    shape.slice(0, shape.length - 1).forEach((dimension) => {
      // Validate the dimension descriptor format
      parseDimDtor(dimension);
    });

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
      type: 'mat',
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
  const format1 = /^CV_([1-9]\d*)(U|S|F)C([1-9]\d*)$/;
  const format2 = /^CV_([1-9]\d*)(U|S|F)$/;
  const format3 = /^(char|int|float|double)$/;
  let result;
  let parsed = { toOrigStr: toOrigStr.bind({ origStr: str }) };
  if (format1.test(str)) {
    result = format1.exec(str);

    const channels = result[3];
    const depth = `${result[1]}${result[2]}`;
    if (channels > 4) {
      // Maximum channel number is 4
      throw new Error(`Invalid Data Cell format of channel: expect number between 1, 4 but get ${channels} from ${str}`);
    }
    if (depth !== '8U' && depth !== '8S' && depth !== '16U' && depth !== '16S'
        && depth !== '32S' && depth !== '32F' && depth !== '64F') {
      throw new Error(`Invalid Data Cell format of depth: ${depth} from ${str}`);
    }
    parsed.format = 'cv';
    parsed.type = depth;
    parsed.channels = parseInt(channels);
    return parsed;
  } else if (format2.test(str)) {
    result = format2.exec(str);

    const depth = `${result[1]}${result[2]}`;
    if (depth !== '8U' && depth !== '8S' && depth !== '16U' && depth !== '16S'
        && depth !== '32S' && depth !== '32F' && depth !== '64F') {
      throw new Error(`Invalid Data Cell format of depth: ${depth} from ${str}`);
    }
    parsed.format = 'cv';
    parsed.type = depth;
    parsed.channels = 1;
    return parsed;
  } else if (format3.test(str)) {
    result = format3.exec(str);
    parsed.format = 'std';
    parsed.type = result[1];
    parsed.channels = 1;
    return parsed;
  } else {
    throw new Error(`Invalid data descriptor format: ${str}`);
  }

  function toOrigStr() {
    return this.origStr;
  }
}

/**
 * Parses the attribute type expression string.
 *
 * The format of attribute type expression please see:
 * https://www.tensorflow.org/versions/r0.12/how_tos/adding_an_op/index.html#attr_types
 *
 * @param {array} str The attribute type expression string to be parsed.
 * @return {object} The parsed object.
 *
 */
function parseAttrType(str) {
  if (typeof str !== 'string') {
    throw new Error(`Invalid attribute type format: ${str}`);
  }
  // TODO: Support for more attribute types
  const format = /^(string|int|float|bool) *= *([^ ].*)/;

  if (!format.test(str)) {
    throw new Error(`Invalid attribute type format: ${str}`);
  }

  const result = format.exec(str);
  const type = result[1];
  const defaultVal = result[2].trim();

  if (type === 'string') {
    if ((defaultVal.length < 2) ||
        !((defaultVal[0] === '\'' && defaultVal[defaultVal.length - 1] === '\'') ||
          (defaultVal[0] === '\"' && defaultVal[defaultVal.length - 1] === '\"'))
    ) {
      throw new Error(`Invalid default value for string: ${defaultVal} from ${str}`);
    }
  } else if (type === 'int') {
    if (!/^(0|-?[1-9]\d*)$/.test(defaultVal)) {
      throw new Error(`Invalid default value for int: ${defaultVal} from ${str}`);
    }
  } else if (type === 'float') {
    if (!/^-?(0|0\.\d+|\.\d+|[1-9]\d*(\.\d+)?)$/.test(defaultVal)) {
      throw new Error(`Invalid default value for float: ${defaultVal} from ${str}`);
    }
  } else if (type === 'bool') {
    if (['0', '1', 'false', 'true', 'False', 'True'].indexOf(defaultVal) === -1) {
      throw new Error(`Invalid default value for bool: ${defaultVal} from ${str}`);
    }
  }

  return {
    type,
    defaultVal
  };
}

/**
 * Given an array and match function, find the index of last matched element in the array.
 * Return -1 if no element is matched.
 *
 * @param {string} arr The array to be matched
 * @param {function} fn The match function
 * @return {number} The index of last matched element; -1 if no element is mathed.
 *
 */
function lastIndexOf(arr, fn) {
  let lastIndex = -1;
  if (!Array.isArray(arr) || !lodash.isFunction(fn)) {
    return lastIndex;
  }
  arr.forEach((elem, index) => {
    if (fn(elem)) {
      lastIndex = index;
    }
  });
  return lastIndex;
}

/**
 * This function will generate a list of square brackets with custom index string.
 *
 * @param {string} template The template string for the generated brackets strings.
 * @param {number} start The start index of the generated brackets string.
 * @param {number} end The end index of the generated brackets string.
 * @example
 *
 * expandArrayAccessor('var_', 3)
 * // => '[var_0][var_1][var_2]'
 *
 * expandArrayAccessor('var_', 1, 3)
 * // => '[var_1][var_2]'
 *
 */
function expandArrayAccessor(template, start, end) {
  if (!lodash.isString(template)) {
    throw new Error(`[expandArrayAccessor] Invalid parameters: template ${template} is not a string`);
  }
  if (!lodash.isInteger(start)) {
    throw new Error(`[expandArrayAccessor] Invalid parameters: start index ${start} is not an integer`);
  }
  if (!lodash.isInteger(end) && end !== undefined) {
    throw new Error(`[expandArrayAccessor] Invalid parameters: end index ${end} is not an integer`);
  }
  return lodash.range(start, end).map((idx) => {
    return `[${template}${idx}]`;
  }).join('');
}

/**
 * This function will generate a list of arguments with the index number as the postfix.
 *
 * @param {string|function} template The template string/function for the generated argument strings.
 * @param {number} start The start index of the generated arguments.
 * @param {number} end The end index of the generated arguments.
 * @example
 *
 * expandArgus('var_', 3)
 * // => 'var_0, var_1, var_2'
 *
 * expandArgus('var_', 1, 3)
 * // => 'var_1, var_2'
 */
function expandArgus(template, start, end) {
  if (!lodash.isString(template) && !lodash.isFunction(template)) {
    throw new Error(`[expandArgus] Invalid parameters: template ${template} is not a string or function`);
  }
  if (!lodash.isInteger(start)) {
    throw new Error(`[expandArgus] Invalid parameters: start index ${start} is not an integer`);
  }
  if (!lodash.isInteger(end) && end !== undefined) {
    throw new Error(`[expandArgus] Invalid parameters: end index ${end} is not an integer`);
  }
  return lodash.range(start, end).map((idx) => {
    if (typeof template === 'function') {
      return template(idx);
    }
    return `${template}${idx}`;
  }).join(', ');
}

/**
 * Generate if/else statements
 *
 * @ param {string} condition An expression that is considered to be either truthy or falsy.
 * @ param {string} truthyStmts Statement that is executed if condition is truthy.
 * @ param {string} falsyStmts Statement that is executed if condition is falsy.
 * @example
 *
 * genIfElse('hello === 10', 'hello++;', 'hello--;')
 * // =>
 * 'if (hello === 10) {
 *     hello++;
 * } else {
 *     hello--;
 * }'
 *
 * genIfElse('hello === 10', 'hello++;')
 * // =>
 * 'if (hello === 10) {
 *     hello++;
 * }'
 */
function genIfElse(condition, truthyStmts, falsyStmts) {
  // Return empty string if condition is non-string or empty string
  if ((!lodash.isString(condition)) || (condition.length === 0)) {
    return '';
  }
  // Return empty string if truthy statements is non-string or empty string
  if ((!lodash.isString(truthyStmts)) || (truthyStmts.length === 0)) {
    return '';
  }

  // TODO: Intent for "if" and "else"
  const indentTruthyStmts = truthyStmts.split('\n').join('\n    ');
  if ((!lodash.isString(falsyStmts)) || (falsyStmts.length === 0)) {
    // Return if statment if falsy statements is non-string or empty string
    return `if (${condition}) {\n    ${indentTruthyStmts}\n}`;
  } else {
    // Return if and else statment if falsy statements is non-empty string
    const indentFalsyStmts = falsyStmts.split('\n').join('\n    ');
    return `if (${condition}) {\n    ${indentTruthyStmts}\n} else {\n    ${indentFalsyStmts}\n}`;
  }
}

module.exports = {
  getTfRank,
  getCvRank,
  parseDimDtor,
  parseDataDtor,
  parseAttrType,
  lastIndexOf,
  expandArrayAccessor,
  expandArgus,
  genIfElse
};

