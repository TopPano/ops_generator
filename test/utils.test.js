import test from 'ava';
import { expect } from 'chai';
import _ from 'lodash';

import {
  VALID_CV_DEPTHS,
  INVALID_CV_DEPTHS,
  VALID_STD_TYPES,
  MAX_CV_CHANNELS
} from './const';
import { testThrownMsg } from './helper';

import utils from '../lib/utils';

const SHAPE_NON_ARRAY_MSG = 'Invalid arguments: no shape array found.';
const SHAPE_EMPTY_ARRAY_MSG = 'Invalid shape format: empty shape array.';
const SHAPE_RANK_ZERO_MULTICHANNEL_MSG = 'Invalid shape format: rank 0 with multichannel is not allowed.';
const SHAPE_INVALID_DIMENSION_MSG = 'Invalid shape format: only accept postive integer or "none" for dimension';

test('getTfRank: calculate the rank number of the shape in tensorflow aspect', t => {
  // Test for non-array input
  [undefined, null, 'CV32', 'int', 3].forEach((shape) => testThrownMsg(t, SHAPE_NON_ARRAY_MSG, utils.getTfRank, shape));
  // Test for empty array
  testThrownMsg(t, SHAPE_EMPTY_ARRAY_MSG, utils.getTfRank, []);
  // Test for valid inputs
  _.range(0, 20).forEach((rawRank) => {
    const rawInput = _.fill(Array(rawRank), 1);
    // Test for channel = 1, expect returned rank equal to raw rank
    VALID_CV_DEPTHS.forEach((depth) => {
      t.is(utils.getTfRank(_.concat(rawInput, `CV_${depth}`)), rawRank);
      t.is(utils.getTfRank(_.concat(rawInput, `CV_${depth}C1`)), rawRank);
    });
    VALID_STD_TYPES.forEach((type) => {
      t.is(utils.getTfRank(_.concat(rawInput, type)), rawRank);
    });
    // Test for multichannel, expect returned rank equal to raw rank + 1
    VALID_CV_DEPTHS.forEach((depth) => {
      _.range(2, MAX_CV_CHANNELS + 1).forEach((channels) => {
        t.is(utils.getTfRank(_.concat(rawInput, `CV_${depth}C${channels}`)), rawRank + 1);
      });
    });
  });
});

test('getCvRank: calculate the rank number of the shape in OpenCV aspect', t => {
  // Test for non-array input
  [undefined, null, 'CV32', 'int', 3].forEach((shape) => testThrownMsg(t, SHAPE_NON_ARRAY_MSG, utils.getCvRank, shape));
  // Test for empty array
  testThrownMsg(t, SHAPE_EMPTY_ARRAY_MSG, utils.getCvRank, []);
  // Test for valid inputs, expect return rank eqaul to raw rank
  _.range(0, 20).forEach((rawRank) => {
    const rawInput = _.fill(Array(rawRank), 1);
    // Test for channel = 1
    VALID_CV_DEPTHS.forEach((depth) => {
      t.is(utils.getCvRank(_.concat(rawInput, `CV_${depth}`)), rawRank);
      t.is(utils.getCvRank(_.concat(rawInput, `CV_${depth}C1`)), rawRank);
    });
    VALID_STD_TYPES.forEach((type) => {
      t.is(utils.getTfRank(_.concat(rawInput, type)), rawRank);
    });
    // Test for multichannel
    VALID_CV_DEPTHS.forEach((depth) => {
      _.range(2, MAX_CV_CHANNELS + 1).forEach((channels) => {
        t.is(utils.getCvRank(_.concat(rawInput, `CV_${depth}C${channels}`)), rawRank);
      });
    });
  });
});

const INVALID_DIM_DTOR_MSG = 'Invalid dimensional descriptor format';
const INVALID_DIM_ZERO_MSG = 'Invalid dimensional descriptor format: dimension should > 0';

function testParsedDimDtorOutput(t, dtor, expectedType, expectedDims) {
  t.deepEqual(utils.parseDimDtor(dtor), {
    type: expectedType,
    dims: expectedDims
  });
}

test('parseDimDtor: parse dimensional descriptor string', t => {
  // Test for invalid dimensional descriptor format
  [undefined, null, [], {}, 0.5, -1, NaN, '', "", 'NONE', 'none1', '1none', 'vector', 'vector:', 'vector:NONE', '1vector:none', 'vector:none1', 'vector:1none'].forEach((dtor) => {
    testThrownMsg(t, `${INVALID_DIM_DTOR_MSG}: ${dtor}`, utils.parseDimDtor, dtor);
  });
  // Test for dimension = 0
  [0, 0.0, '0', 'vector:0'].forEach((dtor) => {
    testThrownMsg(t, `${INVALID_DIM_ZERO_MSG}`, utils.parseDimDtor, dtor);
  })
  // Test for valid inputs
  testParsedDimDtorOutput(t, 'none', 'mat', 'none');
  testParsedDimDtorOutput(t, 'vector:none', 'vector', 'none');
  _.range(0, 20).forEach((power) => {
    const dims = Math.pow(2, power);
    testParsedDimDtorOutput(t, `${dims}`, 'mat', dims);
    testParsedDimDtorOutput(t, `vector:${dims}`, 'vector', dims);
  });
});

const INVALID_DATA_DTOR_MSG = 'Invalid data descriptor format';
const INVALID_DATA_CHANNELS_MSG = 'Invalid Data Cell format of channel: expect number between 1, 4 but get';
const INVALID_DATA_DEPTH_MSG = 'Invalid Data Cell format of depth';

function testInvalidChannelsErrMsg(t, dtor, channels) {
  testThrownMsg(t, `${INVALID_DATA_CHANNELS_MSG} ${channels} from ${dtor}`, utils.parseDataDtor, dtor);
}

function testInvalidDepthErrMsg(t, dtor, depth) {
  testThrownMsg(t, `${INVALID_DATA_DEPTH_MSG}: ${depth} from ${dtor}`, utils.parseDataDtor, dtor);
}

function testParsedDataDtorOutput(t, dtor, expectedFormat, expectedType, expectedChannels) {
  const output = utils.parseDataDtor(dtor);

  expect(output).to.be.an('object');
  expect(output.toOrigStr).to.be.an('function');
  t.is(output.toOrigStr(), dtor);
  t.deepEqual(_.omit(output, 'toOrigStr'), {
    format: expectedFormat,
    type: expectedType,
    channels: expectedChannels
  });
}

test('parseDataDtor: parse data descriptor string', t => {
  // Test for invalid data descriptor format
  [undefined, null, [], {}, NaN, '', "", 'CV_0U', 'CV-10S', 'CV_8s', 'CV_16FC', 'CV_16FC0', 'CV_32SC3s', 'charA', 'aCV_8U', 'aCV_8UC1', 'achar'].forEach((dtor) => {
    testThrownMsg(t, `${INVALID_DATA_DTOR_MSG}: ${dtor}`, utils.parseDataDtor, dtor);
  });

  /* Tests for CvTypes */
  // Test for valid OpenCV depths
  VALID_CV_DEPTHS.forEach((depth) => {
    // Without channel postfix, which means channels = 1
    testParsedDataDtorOutput(t, `CV_${depth}`, 'cv', depth, 1);
    // With channel postfix
    _.range(1, MAX_CV_CHANNELS + 1).forEach((channels) => {
      testParsedDataDtorOutput(t, `CV_${depth}C${channels}`, 'cv', depth, channels);
    });
  });
  // Test for invalid OpenCV depths
  INVALID_CV_DEPTHS.forEach((depth) => {
    // Without channel postfix, which means channels = 1
    testInvalidDepthErrMsg(t, `CV_${depth}`, depth);
    // With channel postfix
    _.range(1, MAX_CV_CHANNELS + 1).forEach((channels) => {
      testInvalidDepthErrMsg(t, `CV_${depth}C${channels}`, depth)
    });
  });
  // Test for invalid OpenCV channels
  _.range(5, 8).forEach((channels) => testInvalidChannelsErrMsg(t, `CV_8UC${channels}`, channels));
  _.range(3, 20).forEach((power) => {
    const channels = Math.pow(2, power);
    testInvalidChannelsErrMsg(t, `CV_8UC${channels}`, channels);
  });

  // Test for supported standard C types
  VALID_STD_TYPES.forEach((type) => testParsedDataDtorOutput(t, type, 'std', type, 1));
});

const INVALID_ATTR_TYPE_MSG = 'Invalid attribute type format';
const INVALID_DEFAULT_VALUE_MSG = 'Invalid default value for';

function testParsedAttrTypeOutput(t, attrType, expectedType, expectedDefaultVal) {
  t.deepEqual(utils.parseAttrType(attrType), {
    type: expectedType,
    defaultVal: expectedDefaultVal
  });
}

// Currently only support: string, int, float, bool
// The list of full attribute types:
// https://www.tensorflow.org/versions/r0.12/how_tos/adding_an_op/index.html#attr_types
test('parseAttrType: parse attribute expression string', t => {
  // Test for invalid input
  [undefined, null, 1, 0.5, -0.5, NaN, '', "", {}, () => {}, 'uint8', 'char',
    'string', ' string', '  string', 'string =', 'int', 'int = ', 'float  =', 'bool  =  '].forEach((attrType) => {
    testThrownMsg(
      t,
      `${INVALID_ATTR_TYPE_MSG}: ${attrType}`,
      utils.parseAttrType,
      attrType
    );
  });
  // Test for invalid default value of string
  ['\'', '\"', '\'\"', '\"\'', 'non-quoted string', '\'word 1\' word2'].forEach((defaultVal) => {
    const attrType = `string = ${defaultVal}`;
    testThrownMsg(
      t,
      `${INVALID_DEFAULT_VALUE_MSG} string: ${defaultVal} from ${attrType}`,
      utils.parseAttrType,
      attrType
    );
  });
  // Test for valid default value of string
  ['\'\'', '\"\"', '\'valid\'', '\"valid\"', '\"\'valid\'\"', '\'\"valid\'\''].forEach((defaultVal) => {
    testParsedAttrTypeOutput(t, `string = ${defaultVal}`, 'string', defaultVal);
  });
  // Test for invalid default value of int
  ['0.5', '.5', '+1', 'characters', '1 2', '00', '0123'].forEach((defaultVal) => {
    const attrType = `int = ${defaultVal}`;
    testThrownMsg(
      t,
      `${INVALID_DEFAULT_VALUE_MSG} int: ${defaultVal} from ${attrType}`,
      utils.parseAttrType,
      attrType
    );
  });
  // Test for valid default value of int
  ['0', '10', '-10'].forEach((defaultVal) => {
    testParsedAttrTypeOutput(t, `int = ${defaultVal}`, 'int', defaultVal);
  });
  // Test for invalid default value of float
  ['+1.0', 'characters', '00', '0123', '00.', '1.0 2.0'].forEach((defaultVal) => {
    const attrType = `float = ${defaultVal}`;
    testThrownMsg(
      t,
      `${INVALID_DEFAULT_VALUE_MSG} float: ${defaultVal} from ${attrType}`,
      utils.parseAttrType,
      attrType
    );
  });
  // Test for valid default value of int
  ['0', '0.0', '0.000', '-0.0', '1.1', '1.111', '-1.111'].forEach((defaultVal) => {
    testParsedAttrTypeOutput(t, `float = ${defaultVal}`, 'float', defaultVal);
  });
  // Test for invalid default value of bool
  ['characters', 'TRUE', 'FALSE', '3', '-1'].forEach((defaultVal) => {
    const attrType = `bool = ${defaultVal}`;
    testThrownMsg(
      t,
      `${INVALID_DEFAULT_VALUE_MSG} bool: ${defaultVal} from ${attrType}`,
      utils.parseAttrType,
      attrType
    );
  });
  // Test for valid default value of bool
  ['0', '1', 'false', 'true', 'False', 'True'].forEach((defaultVal) => {
    testParsedAttrTypeOutput(t, `bool = ${defaultVal}`, 'bool', defaultVal);
  });
});

test('lastIndexOf: given an array and match function, find the index of last matched element in the array', t => {
  // Test for non-array input
  [undefined, null, 1, 0.5, -0.5, NaN, '', "", {}, () => {}].forEach((arr) => t.is(utils.lastIndexOf(arr), -1));
  // Test for non-function input
  [undefined, null, 1, 0.5, -0.5, NaN, '', "", {}, []].forEach((fn) => t.is(utils.lastIndexOf([], fn), -1));

  /* Tests for valid inputs */
  const matchFn = (x) => x === 'vector';
  // Test for empty array
  t.is(utils.lastIndexOf([], () => {}), -1);
  // Test for non-matched array
  t.is(utils.lastIndexOf(_.fill(Array(100), 'VECTOR'), matchFn), -1);
  // Test for fully-matched array
  t.is(utils.lastIndexOf(_.fill(Array(100), 'vector'), matchFn), 99);
  // Test for partially-matched array
  _.range(1, 20).forEach((power) => {
    const length = Math.pow(2, power);
    const arr = _.fill(Array(length), 'VECTOR');
    arr[0] = arr[length - 2] = 'vector';
    t.is(utils.lastIndexOf(arr, matchFn), length - 2);
  });
});

const INVALID_EXPAND_ARRAY_PARAMS_MSG = '[expandArrayAccessor] Invalid parameters';

function testExpandedArrayAccessorOutput(t, template, start, end) {
  t.is(
    utils.expandArrayAccessor(template, start, end),
    _.range(start, end).map((idx) => `[${template}${idx}]`).join('')
 );
}

test('expandArrayAccessor: generate a list of square brackets with custom index string', t => {
  const validTemplate = 'var';

  // Test for invalid template
  [undefined, null, 0, 0.5, -0.5, NaN, {}, [], () => {}].forEach((template) => {
    testThrownMsg(
      t,
      `${INVALID_EXPAND_ARRAY_PARAMS_MSG}: template ${template} is not a string`,
      utils.expandArrayAccessor,
      template
    );
  });
  // Test for non-integer end index
  [undefined, null, '', "", 0.5, -0.5, NaN, {}, [], () => {}].forEach((end) => {
    testThrownMsg(
      t,
      `${INVALID_EXPAND_ARRAY_PARAMS_MSG}: end index ${end} is not an integer`,
      utils.expandArrayAccessor,
      validTemplate,
      end
    );
  });

  /* Tests for valid inputs */
  // Test for inputs without start index
  _.range(0, 10).forEach((power) => {
    const endIdx = Math.pow(2, power);
    testExpandedArrayAccessorOutput(t, validTemplate, endIdx);
    testExpandedArrayAccessorOutput(t, validTemplate, -endIdx);
  });
  // Test for inputs with start index
  _.range(0, 5).forEach((power) => {
    const startIdx = Math.pow(2, power);
    _.range(0, 5).forEach((power) => {
      const endIdx = Math.pow(2, power);
      testExpandedArrayAccessorOutput(t, validTemplate, 0, endIdx);
      testExpandedArrayAccessorOutput(t, validTemplate, 0, -endIdx);
      testExpandedArrayAccessorOutput(t, validTemplate, startIdx, endIdx);
      testExpandedArrayAccessorOutput(t, validTemplate, startIdx, -endIdx);
      testExpandedArrayAccessorOutput(t, validTemplate, -startIdx, endIdx);
      testExpandedArrayAccessorOutput(t, validTemplate, -startIdx, -endIdx);
    });
  });
});

const INVALID_EXPAND_ARGS_PARAMS_MSG = '[expandArgus] Invalid parameters';

function testExpandedArgusOutput(t, template ,start, end) {
  t.is(
    utils.expandArgus(template, start, end),
    _.range(start, end).map((idx) => _.isFunction(template) ? template(idx) : `${template}${idx}`).join(', ')
 );
}

test('expandArgus: generate a list of arguments with the index number as the postfix', t => {
  const validTemplateStr = 'var';
  const validTemplateFn = (x) => `var-${x}`;

  // Test for invalid template
  [undefined, null, 0, 0.5, -0.5, NaN, {}, []].forEach((template) => {
    testThrownMsg(
      t,
      `${INVALID_EXPAND_ARGS_PARAMS_MSG}: template ${template} is not a string or function`,
      utils.expandArgus,
      template
    );
  });
  // Test for non-integer end index
  [undefined, null, '', "", 0.5, -0.5, NaN, {}, [], () => {}].forEach((end) => {
    testThrownMsg(
      t,
      `${INVALID_EXPAND_ARGS_PARAMS_MSG}: end index ${end} is not an integer`,
      utils.expandArgus,
      validTemplateStr,
      end
    );
  });

  /* Tests for valid inputs */
  // Test for inputs without start index
  _.range(0, 10).forEach((power) => {
    const endIdx = Math.pow(2, power);
    testExpandedArgusOutput(t, validTemplateStr, endIdx);
    testExpandedArgusOutput(t, validTemplateFn, endIdx);
    testExpandedArgusOutput(t, validTemplateStr, -endIdx);
    testExpandedArgusOutput(t, validTemplateFn, -endIdx);
  });
  // Test for inputs with start index
  _.range(0, 5).forEach((power) => {
    const startIdx = Math.pow(2, power);
    _.range(0, 5).forEach((power) => {
      const endIdx = Math.pow(2, power);
      testExpandedArgusOutput(t, validTemplateStr, 0, endIdx);
      testExpandedArgusOutput(t, validTemplateFn, 0, endIdx);
      testExpandedArgusOutput(t, validTemplateStr, 0, -endIdx);
      testExpandedArgusOutput(t, validTemplateFn, 0, -endIdx);
      testExpandedArgusOutput(t, validTemplateStr, startIdx, endIdx);
      testExpandedArgusOutput(t, validTemplateFn, startIdx, endIdx);
      testExpandedArgusOutput(t, validTemplateStr, startIdx, -endIdx);
      testExpandedArgusOutput(t, validTemplateFn, startIdx, -endIdx);
      testExpandedArgusOutput(t, validTemplateStr, -startIdx, endIdx);
      testExpandedArgusOutput(t, validTemplateFn, -startIdx, endIdx);
      testExpandedArgusOutput(t, validTemplateStr, -startIdx, -endIdx);
      testExpandedArgusOutput(t, validTemplateFn, -startIdx, -endIdx);
    });
  });
});

const NO_TRUTHY_STMT_MSG = 'genIfElse: Truthy statements must be a non-empty string, but find';

test('genIfElse: Generate if/else statements', t => {
  // Tests for condition is non-string or empty string
  [undefined, null, 10, 0.5, [], {}, () => {}, ''].forEach((condition) => {
    t.is(utils.genIfElse(condition), '');
  });
  // Tests for truthyStmts is non-string or empty string
  [undefined, null, 10, 0.5, [], {}, () => {}, ''].forEach((truthyStmts) => {
    t.is(utils.genIfElse('true', truthyStmts), '');
    // testThrownMsg(t, `${NO_TRUTHY_STMT_MSG}: ${truthyStmts}`, utils.genIfElse, 'true', truthyStmts);
  });
  // Tests for falsyStmts is non-string or empty string
  [undefined, null, 10, 0.5, [], {}, () => {}, ''].forEach((falsyStmts) => {
    t.is(
      utils.genIfElse('true', 'a++;', falsyStmts),
      'if (true) {\n    a++;\n}'
    );
  });
  // Tests for condition, truthyStmts and falsyStmts are non-empty string
  t.is(
    utils.genIfElse('a == b', 'a++;', 'b++;'),
    'if (a == b) {\n    a++;\n} else {\n    b++;\n}'
  );
  t.is(
    utils.genIfElse('a == b', 'a++;\nc++;', 'b++;\nd++;'),
    'if (a == b) {\n    a++;\n    c++;\n} else {\n    b++;\n    d++;\n}'
  );
  t.is(
    utils.genIfElse('a == b', '  a++;\n  c++;', '  b++;\n  d++;'),
    'if (a == b) {\n      a++;\n      c++;\n} else {\n      b++;\n      d++;\n}'
  );
});
