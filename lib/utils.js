'use strict';
const lodash = require('lodash');
const types = require('./types');


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
 * @param {number} [start=0] The start index of the generated brackets string.
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
  if (!lodash.isInteger(start) && !lodash.isInteger(end)) {
    throw new Error(`[expandArrayAccessor] Invalid parameters: end index ${start} is not an integer`);
  }
  return lodash.range(start, end).map((idx) => {
    return `[${template}${idx}]`;
  }).join('');
}

/**
 * This function will generate a list of arguments with the index number as the postfix.
 *
 * @param {string|function} template The template string/function for the generated argument strings.
 * @param {number} [start=0] The start index of the generated arguments.
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
  if (!lodash.isInteger(start) && !lodash.isInteger(end)) {
    throw new Error(`[expandArgus] Invalid parameters: end index ${start} is not an integer`);
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
  parseAttrType,
  lastIndexOf,
  expandArrayAccessor,
  expandArgus,
  genIfElse
};

