import test from 'ava';
import { expect } from 'chai';
import sinon from 'sinon';
import _ from 'lodash';

import {
  VALID_CV_DEPTHS,
  INVALID_CV_DEPTHS,
  VALID_STD_TYPES,
  MAX_CV_CHANNELS
} from './const';
import { testThrownMsg } from './helper';
import { parseShape, getTfRank, getCvRank, parseDimDtor, parseDataDtor } from '../lib/shape';


test('getTfRank: calculate the rank number of the shape in tensorflow aspect', t => {
  // Test for non-array input
  [undefined, null, 'CV32', 'int', 3].forEach((shape) => testThrownMsg(t, SHAPE_NON_ARRAY_MSG, getTfRank, shape));
  // Test for empty array
  testThrownMsg(t, SHAPE_EMPTY_ARRAY_MSG, getTfRank, []);
  // Test for valid inputs
  _.range(0, 20).forEach((rawRank) => {
    const rawInput = _.fill(Array(rawRank), 1);
    // Test for channel = 1, expect returned rank equal to raw rank
    VALID_CV_DEPTHS.forEach((depth) => {
      t.is(getTfRank(_.concat(rawInput, `CV_${depth}`)), rawRank);
      t.is(getTfRank(_.concat(rawInput, `CV_${depth}C1`)), rawRank);
    });
    VALID_STD_TYPES.forEach((type) => {
      t.is(getTfRank(_.concat(rawInput, type)), rawRank);
    });
    // Test for multichannel, expect returned rank equal to raw rank + 1
    VALID_CV_DEPTHS.forEach((depth) => {
      _.range(2, MAX_CV_CHANNELS + 1).forEach((channels) => {
        t.is(getTfRank(_.concat(rawInput, `CV_${depth}C${channels}`)), rawRank + 1);
      });
    });
  });
});

test('getCvRank: calculate the rank number of the shape in OpenCV aspect', t => {
  // Test for non-array input
  [undefined, null, 'CV32', 'int', 3].forEach((shape) => testThrownMsg(t, SHAPE_NON_ARRAY_MSG, getCvRank, shape));
  // Test for empty array
  testThrownMsg(t, SHAPE_EMPTY_ARRAY_MSG, getCvRank, []);
  // Test for valid inputs, expect return rank eqaul to raw rank
  _.range(0, 20).forEach((rawRank) => {
    const rawInput = _.fill(Array(rawRank), 1);
    // Test for channel = 1
    VALID_CV_DEPTHS.forEach((depth) => {
      t.is(getCvRank(_.concat(rawInput, `CV_${depth}`)), rawRank);
      t.is(getCvRank(_.concat(rawInput, `CV_${depth}C1`)), rawRank);
    });
    VALID_STD_TYPES.forEach((type) => {
      t.is(getTfRank(_.concat(rawInput, type)), rawRank);
    });
    // Test for multichannel
    VALID_CV_DEPTHS.forEach((depth) => {
      _.range(2, MAX_CV_CHANNELS + 1).forEach((channels) => {
        t.is(getCvRank(_.concat(rawInput, `CV_${depth}C${channels}`)), rawRank);
      });
    });
  });
});

const INVALID_DIM_DTOR_MSG = 'Invalid dimensional descriptor format';
const INVALID_DIM_ZERO_MSG = 'Invalid dimensional descriptor format: dimension should > 0';

function testParsedDimDtorOutput(t, dtor, expectedType, expectedDims) {
  t.deepEqual(parseDimDtor(dtor), {
    type: expectedType,
    dims: expectedDims
  });
}

test('parseDimDtor: parse dimensional descriptor string', t => {
  // Test for invalid dimensional descriptor format
  [undefined, null, [], {}, 0.5, -1, NaN, '', "", 'NONE', 'none1', '1none', 'vector', 'vector:', 'vector:NONE', '1vector:none', 'vector:none1', 'vector:1none'].forEach((dtor) => {
    testThrownMsg(t, `${INVALID_DIM_DTOR_MSG}: ${dtor}`, parseDimDtor, dtor);
  });
  // Test for dimension = 0
  [0, 0.0, '0', 'vector:0'].forEach((dtor) => {
    testThrownMsg(t, `${INVALID_DIM_ZERO_MSG}`, parseDimDtor, dtor);
  })
  // Test for valid inputs
  testParsedDimDtorOutput(t, 'none', 'Mat', 'none');
  testParsedDimDtorOutput(t, 'vector:none', 'vector', 'none');
  _.range(0, 20).forEach((power) => {
    const dims = Math.pow(2, power);
    testParsedDimDtorOutput(t, `${dims}`, 'Mat', dims);
    testParsedDimDtorOutput(t, `vector:${dims}`, 'vector', dims);
  });
});

const INVALID_DATA_DTOR_MSG = 'Invalid data descriptor format';
const INVALID_DATA_DTOR_STATIC_MAT_MSG = 'Invalid data descriptor format: static Mat type does not support multichannels';
const INVALID_DATA_CHANNELS_MSG = 'Invalid Data Cell format of channel: expect number between 1, 4 but get';
const INVALID_DATA_DEPTH_MSG = 'Invalid Data Cell format of depth';

function testInvalidChannelsErrMsg(t, dtor, channels) {
  testThrownMsg(t, `${INVALID_DATA_CHANNELS_MSG} ${channels} from ${dtor}`, parseDataDtor, dtor);
}

function testInvalidDepthErrMsg(t, dtor, depth) {
  testThrownMsg(t, `${INVALID_DATA_DEPTH_MSG}: ${depth} from ${dtor}`, parseDataDtor, dtor);
}

function testParsedDataDtorOutputCv(t, dtor, expectedFormat, expectedCType, expectedDType, expectedChannels) {
  const output = parseDataDtor(dtor);

  expect(output).to.be.an('object');
  expect(output.toString).to.be.an('function');
  expect(output.accessor).to.be.an('function');
  t.is(output.toString(), dtor.split(':')[0]);
  t.deepEqual(_.omit(output, ['toString', 'accessor'] ), {
    format: expectedFormat,
    ctype: expectedCType,
    dtype: expectedDType,
    channels: expectedChannels
  });
}

function testParsedDataDtorOutputStd(t, dtor, expectedFormat, expectedDType, expectedChannels) {
  const output = parseDataDtor(dtor);

  expect(output).to.be.an('object');
  expect(output.toString).to.be.an('function');
  t.is(output.toString(), dtor);
  t.deepEqual(_.omit(output, 'toString'), {
    format: expectedFormat,
    dtype: expectedDType,
    channels: expectedChannels
  });
}

test('parseDataDtor: parse data descriptor string', t => {
  // Test for invalid data descriptor format
  [undefined, null, [], {}, NaN, '', "", 'CV_0U', 'CV-10S', 'CV_8s', 'CV_16FC', 'CV_16FC0', 'CV_32SC3s', 'charA', 'aCV_8U', 'aCV_8UC1', 'achar'].forEach((dtor) => {
    testThrownMsg(t, `${INVALID_DATA_DTOR_MSG}: ${dtor}`, parseDataDtor, dtor);
  });
  testThrownMsg(t, INVALID_DATA_DTOR_STATIC_MAT_MSG, parseDataDtor, 'CV_32FC2:Vec');
  testThrownMsg(t, INVALID_DATA_DTOR_STATIC_MAT_MSG, parseDataDtor, 'CV_32FC2:Matx');

  /* Tests for CvTypes */
  // Test for valid OpenCV depths
  VALID_CV_DEPTHS.forEach((depth) => {
    // Without channel postfix, which means channels = 1
    testParsedDataDtorOutputCv(t, `CV_${depth}`, 'cv', 'Mat', depth, 1);
    testParsedDataDtorOutputCv(t, `CV_${depth}:Matx`, 'cv', 'Matx', depth, 1);
    // With channel postfix
    _.range(1, MAX_CV_CHANNELS + 1).forEach((channels) => {
      testParsedDataDtorOutputCv(t, `CV_${depth}C${channels}`, 'cv', 'Mat', depth, channels);
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
  VALID_STD_TYPES.forEach((type) => testParsedDataDtorOutputStd(t, type, 'std', type, 1));
});

const SHAPE_NON_ARRAY_MSG = 'Invalid arguments: no shape array found.';
const SHAPE_EMPTY_ARRAY_MSG = 'Invalid shape format: empty shape array.';
const SHAPE_RANK_ZERO_MULTICHANNEL_MSG = 'Invalid shape format: rank 0 with multichannel is not allowed.';
const SHAPE_INVALID_DIMENSION_MSG = 'Invalid shape format: only accept postive integer or "none" for dimension';
const SHAPE_INVALID_DIM_DYNAMIC_MSG = 'Invalid shape format: dynamic dimension size is not allowed for static Mat type';
const SHAPE_SCALAR_CV_MSG = 'Invalid shape format: scalar of OpenCV type is not allowed';
const SHAPE_MAT_PRIM_MSG = 'Invalid shape format: Mat of primary type is not allowed';
const SHAPE_VECTOR_CV_MSG = 'Invalid shape format: vector of OpenCV type is not allowed';
const SHAPE_MAT_VECTOR_MSG = 'Invalid shape format: Mat of vector or vector of Mat of vector is not allowed';
const SHAPE_VEC_ONE_DIM_MSG = 'Invalid shape format: Vec type should be one-dimensional';

function testParsedShape(
  t,
  shape,
  expectedTfRank,
  expectedCvRank,
  expectedVecDims,
  expectedMatDims,
  expectedMatDimDtorArr,
  expectedType,
  expectedVarMatDecStr,
  expectedVarDecStr
) {
  const parsedShape = parseShape(shape);
  const result = _.omit(parsedShape, ['dataDtor', 'dimDtorArr']);

  let expectedResult = {
    tfRank: expectedTfRank,
    cvRank: expectedCvRank,
    vecDims: expectedVecDims,
    matDims: expectedMatDims,
    matDimDtorArr: expectedMatDimDtorArr,
    type: expectedType,
    varDecStr: expectedVarDecStr
  };
  if (expectedVarMatDecStr) {
    expectedResult.varMatDecStr = expectedVarMatDecStr;
  }
  t.deepEqual(result, expectedResult);
}

test('parseShape: parse shape array', t => {
  // Tests for non-array input
  [undefined, null, {}, 1, NaN, true, '', () => {}].forEach(shape => {
    testThrownMsg(t, SHAPE_NON_ARRAY_MSG, parseShape, shape)
  });

  // Test for empty array
  testThrownMsg(t, SHAPE_EMPTY_ARRAY_MSG, parseShape, []);
  // Tests for scalar of primary type
  testParsedShape(t, ['int'], 0, 0, 0, 0, [], 'SCALAR', null, 'int');
  testParsedShape(t, ['double'], 0, 0, 0, 0, [], 'SCALAR', null, 'double');

  // Tests for scalar of cv type
  testThrownMsg(t, SHAPE_SCALAR_CV_MSG, parseShape, ['CV_8U'])
  testThrownMsg(t, SHAPE_SCALAR_CV_MSG, parseShape, ['CV_8UC2'])

  // Tests for mat of primary type
  testThrownMsg(t, SHAPE_MAT_PRIM_MSG, parseShape, ['10', 'int']);
  testThrownMsg(t, SHAPE_MAT_PRIM_MSG, parseShape, ['none', '10', 'double']);

  // Tests for mat of cv type
  testParsedShape(t, ['none', 'CV_16S'], 1, 1, 0, 1, [{type: 'Mat', dims: 'none'}], 'MAT', 'Mat', 'Mat');
  testParsedShape(t, ['none', 'CV_8UC2'], 2, 1, 0, 1, [{type: 'Mat', dims: 'none'}], 'MAT', 'Mat', 'Mat');
  testParsedShape(t, ['3', 'none', 'CV_16UC3'], 3, 2, 0, 2, [
    {type: 'Mat', dims: 3},
    {type: 'Mat', dims: 'none'}
  ], 'MAT', 'Mat', 'Mat');
  testParsedShape(t, ['4', 'CV_16S:Vec'], 1, 1, 0, 1, [{type: 'Mat', dims: 4}], 'MAT', 'Vec<int16_t, 4>',
                  'Vec<int16_t, 4>');
  testParsedShape(t, ['4', 'CV_32F:Matx'], 1, 1, 0, 1, [{type: 'Mat', dims: 4}], 'MAT', 'Matx<float, 1, 4>',
                  'Matx<float, 1, 4>');
  testParsedShape(t, ['4', '4', 'CV_32F:Matx'], 2, 2, 0, 2, [
    {type: 'Mat', dims: 4},
    {type: 'Mat', dims: 4}
  ], 'MAT', 'Matx<float, 4, 4>', 'Matx<float, 4, 4>');
  testThrownMsg(t, SHAPE_VEC_ONE_DIM_MSG, parseShape, ['3', '3', 'CV_32F:Vec']);
  testThrownMsg(t, SHAPE_INVALID_DIM_DYNAMIC_MSG, parseShape, ['none', '3', 'CV_32F:Vec']);
  testThrownMsg(t, SHAPE_INVALID_DIM_DYNAMIC_MSG, parseShape, ['none', '3', 'CV_32F:Matx']);

  // Tests for vector of primary type
  testParsedShape(t, ['vector:none', 'double'], 1, 1, 1, 0, [], 'VEC_OF_PRIM', null, 'vector<double>');
  testParsedShape(t, ['vector:20', 'vector:10', 'char'], 2, 2, 2, 0, [], 'VEC_OF_PRIM', null,
                  'vector<vector<char>>');

  // Tests for vector of cv type
  testThrownMsg(t, SHAPE_VECTOR_CV_MSG, parseShape, ['vector:none', 'CV_32S']);
  testThrownMsg(t, SHAPE_VECTOR_CV_MSG, parseShape, ['vector:none', 'CV_32SC1']);
  testThrownMsg(t, SHAPE_VECTOR_CV_MSG, parseShape, ['vector:3', 'vector:10', 'CV_32SC3']);

  // Tests for vector of Mat of primary types
  testThrownMsg(t, SHAPE_MAT_PRIM_MSG, parseShape, ['vector:none', '10', 'int']);
  testThrownMsg(t, SHAPE_MAT_PRIM_MSG, parseShape, ['vector:30', '30', '10', 'char']);

  // Tests for vector of Mat of cv types
  testParsedShape(t, ['vector:none', '100', 'CV_8S'], 2, 2, 1, 1, [{type: 'Mat', dims: 100}], 'VEC_OF_MAT',
                  'Mat', 'vector<Mat>');
  testParsedShape(t, ['vector:none', '100', 'CV_8SC1'], 2, 2, 1, 1, [{type: 'Mat', dims: 100}], 'VEC_OF_MAT',
                  'Mat', 'vector<Mat>');
  testParsedShape(t, ['vector:none', '100', 'CV_8SC3'], 3, 2, 1, 1, [{type: 'Mat', dims: 100}], 'VEC_OF_MAT',
                  'Mat', 'vector<Mat>');

  // Tests for mat of vector
  testThrownMsg(t, SHAPE_MAT_VECTOR_MSG, parseShape, ['10', 'vector:10', 'CV_16U']);
  testThrownMsg(t, SHAPE_MAT_VECTOR_MSG, parseShape, ['vector:none', '10', 'vector:10', 'CV_16U']);
});

