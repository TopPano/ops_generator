import test from 'ava';
import rewire from 'rewire';
import sinon from 'sinon';
import Mustache from 'mustache';
import _ from 'lodash';

import {
  VALID_CV_DEPTHS,
  VALID_STD_TYPES,
  MAX_CV_CHANNELS
} from './const';
import { testThrownMsg } from './helper';

import utils from '../lib/utils';
import template from '../lib/template';

const generator = rewire('../lib/generator');
const parseShape = generator.__get__('parseShape');
const ascendingId = generator.__get__('ascendingId');
const lowerAndSnake = generator.__get__('lowerAndSnake');
const registerOpInput = generator.__get__('registerOpInput');
const registerOpInputFn = generator.__get__('registerOpInputFn');
const registerOpOutput = generator.__get__('registerOpOutput');
const registerOpOutputFn = generator.__get__('registerOpOutputFn');
const opAttributes = generator.__get__('opAttributes');
const registerOpAttrFn = generator.__get__('registerOpAttrFn');
const getAttributesFn = generator.__get__('getAttributesFn');
const declareAttributesFn = generator.__get__('declareAttributesFn');
const registerOpShape = generator.__get__('registerOpShape');
const registerOpShapeFn = generator.__get__('registerOpShapeFn');
const computeInput = generator.__get__('computeInput');
const computeInputFn = generator.__get__('computeInputFn');
const computeExecute = generator.__get__('computeExecute');
const computeExecuteFn = generator.__get__('computeExecuteFn');
const computeOutput = generator.__get__('computeOutput');
const computeOutputFn = generator.__get__('computeOutputFn');
const _render = generator.__get__('_render');


const SHAPE_NON_ARRAY_MSG = 'Invalid arguments: no shape array found.';
const SHAPE_EMPTY_ARRAY_MSG = 'Invalid shape format: empty shape array.';
const SHAPE_INVALID_VECTOR_MSG = 'Invalid shape format: Mat of vector or vector of Mat of vector is not allowed';

function testParsedShape(
  t,
  shape,
  expectedTfRank,
  expectedCvRank,
  expectedVecDims,
  expectedMatDims,
  expectedType,
  expectedTypeDecStr
) {
  const parseDataDtor = sinon.spy(utils, 'parseDataDtor');
  const parseDimDtor = sinon.spy(utils, 'parseDimDtor');
  const parsedShape = parseShape(shape);
  const result = _.omit(parsedShape, ['dataDtor', 'dimDtorArr']);

  t.true(parseDataDtor.called);
  if (shape.length > 1) {
    t.true(parseDimDtor.called);
  } else {
    t.false(parseDimDtor.called);
  }
  t.deepEqual(result, {
    tfRank: expectedTfRank,
    cvRank: expectedCvRank,
    vecDims: expectedVecDims,
    matDims: expectedMatDims,
    type: expectedType,
    typeDecStr: expectedTypeDecStr
  });

  parseDataDtor.restore();
  parseDimDtor.restore();
}

test('parseShape: parse shape array', t => {
  // Tests for non-array input
  [undefined, null, {}, 1, NaN, true, '', () => {}].forEach(shape => {
    testThrownMsg(t, SHAPE_NON_ARRAY_MSG, parseShape, shape)
  });
  // Test for empty array
  testThrownMsg(t, SHAPE_EMPTY_ARRAY_MSG, parseShape, []);
  // Tests for data descriptor only (rank = 0)
  testParsedShape(t, ['CV_8U'], 0, 0, 0, 0, 'MAT', 'Mat');
  testParsedShape(t, ['int'], 0, 0, 0, 0, 'MAT', 'Mat');
  // Tests for mat type
  testParsedShape(t, ['none', 'CV_16S'], 1, 1, 0, 1, 'MAT', 'Mat');
  testParsedShape(t, ['none', 'CV_8UC2'], 2, 1, 0, 1, 'MAT', 'Mat');
  testParsedShape(t, ['3', 'none', 'float'], 2, 2, 0, 2, 'MAT', 'Mat');
  // Tests for vector of primary type
  testParsedShape(t, ['vector:none', 'CV_32S'], 1, 1, 1, 0, 'VEC_OF_PRIM', 'vector<int32_t>');
  testParsedShape(t, ['vector:none', 'CV_32SC1'], 1, 1, 1, 0, 'VEC_OF_PRIM', 'vector<int32_t>');
  testParsedShape(t, ['vector:3', 'vector:10', 'CV_32F'], 2, 2, 2, 0, 'VEC_OF_PRIM', 'vector<vector<float>>');
  testParsedShape(t, ['vector:none', 'double'], 1, 1, 1, 0, 'VEC_OF_PRIM', 'vector<double>');
  testParsedShape(t, ['vector:20', 'vector:10', 'char'], 2, 2, 2, 0, 'VEC_OF_PRIM', 'vector<vector<char>>');
  // Test for vector of Mat
  testParsedShape(t, ['vector:none', '100', 'CV_8S'], 2, 2, 1, 1, 'VEC_OF_MAT', 'vector<Mat>');
  testParsedShape(t, ['vector:none', '100', 'CV_8SC1'], 2, 2, 1, 1, 'VEC_OF_MAT', 'vector<Mat>');
  testParsedShape(t, ['vector:none', '100', 'CV_8SC3'], 3, 2, 1, 1, 'VEC_OF_MAT', 'vector<Mat>');
  testParsedShape(t, ['vector:10', '3', 'double'], 2, 2, 1, 1, 'VEC_OF_MAT', 'vector<Mat>');
  testParsedShape(t, ['vector:none', '3', '10', 'int'], 3, 3, 1, 2, 'VEC_OF_MAT', 'vector<Mat>');
  // Tests for vector of CV
  testParsedShape(t, ['vector:3', 'CV_64FC2'], 2, 1, 1, 0, 'VEC_OF_CV', 'vector<Vec<double, 2>>');
  testParsedShape(t, ['vector:none', 'vector:10', 'CV_8UC3'], 3, 2, 2, 0, 'VEC_OF_CV', 'vector<vector<Vec<uint8_t, 3>>>');
  // Test for mat of vector
  testThrownMsg(t, SHAPE_INVALID_VECTOR_MSG, parseShape, ['10', 'vector:10', 'CV_16U']);
  testThrownMsg(t, SHAPE_INVALID_VECTOR_MSG, parseShape, ['vector:none', '10', 'vector:10', 'CV_16U']);
});

test('ascendingId: given objects a and b, return a.id - b.id', t => {
  t.deepEqual(ascendingId({}, {}), NaN);
  t.deepEqual(ascendingId({}, { id: 3 }), NaN);
  t.deepEqual(ascendingId({ id: 'str' }, { id: 3 }), NaN);
  t.deepEqual(ascendingId({ id: 0 }, { id: 3 }), -3);
  t.deepEqual(ascendingId({ id: 3 }, { id: 0 }), 3);
  t.deepEqual(ascendingId({ id: 100 }, { id: 100 }), 0);
});

test('lowerAndSnake: convert string to snake_case', t => {
  t.is(lowerAndSnake(), '');
  t.is(lowerAndSnake(''), '');
  t.is(lowerAndSnake('.'), '');
  t.is(lowerAndSnake('_'), '');
  t.is(lowerAndSnake('name'), 'name');
  t.is(lowerAndSnake('myName'), 'my_name');
  t.is(lowerAndSnake('MyName'), 'my_name');
  t.is(lowerAndSnake('CAPiTaL'), 'ca_pi_ta_l');
  t.is(lowerAndSnake('123numBer'), '123num_ber');
  t.is(lowerAndSnake('__under__score__'), 'under_score');
});

test('registerOpInput: convert inputs and inputoutputs of parsed operations metadata to array', t => {
  // Tests for empty array result
  t.deepEqual(registerOpInput(), []);
  t.deepEqual(registerOpInput({}), []);
  t.deepEqual(registerOpInput({ inputs: [] }), []);
  t.deepEqual(registerOpInput({ inputoutputs: [] }), []);
  // Test for inputs only
  t.deepEqual(registerOpInput({
    inputs: [
      { id: 1, name: 'input1', typeFormat: 'std', type: 'float' }
    ]
  }), [
    { name: 'input1', type: 'float32' }
  ]);
  // Test for inputoutputs only
  t.deepEqual(registerOpInput({
    inputoutputs: [
      { id: 1, name: 'inputoutput1', typeFormat: 'cv', type: '8S' }
    ]
  }), [
    { name: 'inputoutput1_in', type: 'int8' }
  ]);
  // Tests for inputs and inputoutputs
  t.deepEqual(registerOpInput({
    inputs: [
      { id: 1, name: 'input1', typeFormat: 'std', type: 'float' }
    ],
    inputoutputs: [
      { id: 3, name: 'inputoutput3', typeFormat: 'cv', type: '8S' }
    ]
  }), [
    { name: 'input1', type: 'float32' },
    { name: 'inputoutput3_in', type: 'int8' }
  ]);
  t.deepEqual(registerOpInput({
    inputs: [
      { id: 3, name: 'input3', typeFormat: 'cv', type: '8S' }
    ],
    inputoutputs: [
      { id: 1, name: 'inputoutput1', typeFormat: 'std', type: 'float' }
    ]
  }), [
    { name: 'inputoutput1_in', type: 'float32' },
    { name: 'input3', type: 'int8' }
  ]);
  t.deepEqual(registerOpInput({
    inputs: [
      { id: 3, name: 'input3', typeFormat: 'cv', type: '8S' },
      { id: 2, name: 'input2', typeFormat: 'std', type: 'int' }
    ],
    inputoutputs: [
      { id: 4, name: 'inputoutput4', typeFormat: 'std', type: 'double' },
      { id: 1, name: 'inputoutput1', typeFormat: 'cv', type: '32F' }
    ]
  }), [
    { name: 'inputoutput1_in', type: 'float32' },
    { name: 'input2', type: 'int32' },
    { name: 'input3', type: 'int8' },
    { name: 'inputoutput4_in', type: 'float64' }
  ]);
});

test('registerOpInputFn: return string of an input', t => {
  t.is(
    registerOpInputFn.bind({ name: 'input', type: 'int16'})(),
    '.Input("input: int16")'
  );
  t.is(
    registerOpInputFn.bind({ name: 'input', type: 'float32'})(),
    '.Input("input: float32")'
  );
});

test('registerOpOutput: convert outputs and inputoutputs of parsed operations metadata to array', t => {
  // Tests for empty array result
  t.deepEqual(registerOpOutput(), []);
  t.deepEqual(registerOpOutput({}), []);
  t.deepEqual(registerOpOutput({ outputs: [] }), []);
  t.deepEqual(registerOpOutput({ inputoutputs: [] }), []);
  // Test for output only
  t.deepEqual(registerOpOutput({
    outputs: [
      { id: 1, name: 'output1', typeFormat: 'std', type: 'float' }
    ]
  }), [
    { name: 'output1', type: 'float32' }
  ]);
  // Test for inputoutputs only
  t.deepEqual(registerOpOutput({
    inputoutputs: [
      { id: 1, name: 'inputoutput1', typeFormat: 'cv', type: '8S' }
    ]
  }), [
    { name: 'inputoutput1', type: 'int8' }
  ]);
  // Tests for both output and inputoutputs
  t.deepEqual(registerOpOutput({
    outputs: [
      { id: 1, name: 'output1', typeFormat: 'std', type: 'float' }
    ],
    inputoutputs: [
      { id: 3, name: 'inputoutput3', typeFormat: 'cv', type: '8S' }
    ]
  }), [
    { name: 'output1', type: 'float32' },
    { name: 'inputoutput3', type: 'int8' }
  ]);
  t.deepEqual(registerOpOutput({
    outputs: [
      { id: 3, name: 'output3', typeFormat: 'cv', type: '8S' }
    ],
    inputoutputs: [
      { id: 1, name: 'inputoutput1', typeFormat: 'std', type: 'float' }
    ]
  }), [
    { name: 'inputoutput1', type: 'float32' },
    { name: 'output3', type: 'int8' }
  ]);
  t.deepEqual(registerOpOutput({
    outputs: [
      { id: 3, name: 'output3', typeFormat: 'cv', type: '8S' },
      { id: 2, name: 'output2', typeFormat: 'std', type: 'int' }
    ],
    inputoutputs: [
      { id: 4, name: 'inputoutput4', typeFormat: 'std', type: 'double' },
      { id: 1, name: 'inputoutput1', typeFormat: 'cv', type: '32F' }
    ]
  }), [
    { name: 'inputoutput1', type: 'float32' },
    { name: 'output2', type: 'int32' },
    { name: 'output3', type: 'int8' },
    { name: 'inputoutput4', type: 'float64' }
  ]);
});

test('registerOpOutputFn: return string of an output', t => {
  t.is(
    registerOpOutputFn.bind({ name: 'output', type: 'int16'})(),
    '.Output("output: int16")'
  );
  t.is(
    registerOpOutputFn.bind({ name: 'output', type: 'float32'})(),
    '.Output("output: float32")'
  );
});

test('opAttributes: convert attributes of parsed operations metadata to array', t => {
  // Tests for empty array result
  t.deepEqual(opAttributes(), []);
  t.deepEqual(opAttributes({}), []);
  t.deepEqual(opAttributes({ attributes: [] }), []);
  // Test for single-element attributes
  t.deepEqual(opAttributes({
    attributes: [
      { id: 1, name: 'attr1', type: 'float', defaultVal: '1.0'}
    ]
  }), [
    { name: 'attr1', type: 'float', defaultVal: '1.0'}
  ]);
  // Tests for multi-elements attributes
  t.deepEqual(opAttributes({
    attributes: [
      { id: 0, name: 'attr0', type: 'int', defaultVal: '0'},
      { id: 1, name: 'attr1', type: 'float', defaultVal: '1.0'}
    ]
  }), [
    { name: 'attr0', type: 'int', defaultVal: '0'},
    { name: 'attr1', type: 'float', defaultVal: '1.0'}
  ]);
  t.deepEqual(opAttributes({
    attributes: [
      { id: 3, name: 'attr3', type: 'int', defaultVal: '0'},
      { id: 10, name: 'attr10', type: 'float', defaultVal: '10.0'},
      { id: 0, name: 'attr0', type: 'bool', defaultVal: 'true'}
    ]
  }), [
    { name: 'attr0', type: 'bool', defaultVal: 'true'},
    { name: 'attr3', type: 'int', defaultVal: '0'},
    { name: 'attr10', type: 'float', defaultVal: '10.0'}
  ]);
});

test('registerOpAttrFn: return string of an attribute', t => {
  // Tests for input with default value
  t.is(
    registerOpAttrFn.bind({ name: 'attr', type: 'int', defaultVal: '0'})(),
    '.Attr("attr: int = 0")'
  );
  t.is(
    registerOpAttrFn.bind({ name: 'attr', type: 'float', defaultVal: '10.0'})(),
    '.Attr("attr: float = 10.0")'
  );
  t.is(
    registerOpAttrFn.bind({ name: 'attr', type: 'string', defaultVal: '"string"'})(),
    '.Attr("attr: string = "string"")'
  );
  // XXX: should we use enforce user specifying default values?
  // Tests for input without default value
  t.is(
    registerOpAttrFn.bind({ name: 'attr', type: 'int'})(),
    '.Attr("attr: int")'
  );
  t.is(
    registerOpAttrFn.bind({ name: 'attr', type: 'bool'})(),
    '.Attr("attr: bool")'
  );
  // Tests for input with name needs to be coverted to snake_case
  t.is(
    registerOpAttrFn.bind({ name: 'attriBute', type: 'int', defaultVal: '0'})(),
    '.Attr("attri_bute: int = 0")'
  );
  t.is(
    registerOpAttrFn.bind({ name: 'AttribuTe', type: 'float', defaultVal: '10.0'})(),
    '.Attr("attribu_te: float = 10.0")'
  );
});

test('getAttributesFn: return string of getting an attribute', t => {
  t.is(
    getAttributesFn.bind({ name: 'attr' })(),
    'OP_REQUIRES_OK(context, context->GetAttr("attr", &attr_));'
  );
  t.is(
    getAttributesFn.bind({ name: 'attriBute' })(),
    'OP_REQUIRES_OK(context, context->GetAttr("attri_bute", &attriBute_));'
  );
  t.is(
    getAttributesFn.bind({ name: '_attribuTe0' })(),
    'OP_REQUIRES_OK(context, context->GetAttr("attribu_te0", &_attribuTe0_));'
  );
});

test('declareAttributesFn: return string of declaraing a attribute', t => {
  t.is(
    declareAttributesFn.bind({ type: 'int', name: 'attr' })(),
    'int      attr_;'
  );
  t.is(
    declareAttributesFn.bind({ type: 'float', name: 'attr' })(),
    'float      attr_;'
  );
});

test('registerOpShape: convert shapes of outputs and inputoutputs of parsed operations metadata to array', t => {
  // Tests for empty array result
  t.deepEqual(registerOpShape(), []);
  t.deepEqual(registerOpShape({}), []);
  t.deepEqual(registerOpShape({ outputs: [] }), []);
  t.deepEqual(registerOpShape({ inputoutputs: [] }), []);
  // Test for outputs only
  t.deepEqual(registerOpShape({
    outputs: [
      { id: 1, shape: ['CV_8U'] }
    ]
  }), [
    { regIdx: 0, shape: ['CV_8U'] }
  ]);
  // Test for inputoutputs only
  t.deepEqual(registerOpShape({
    inputoutputs: [
      { id: 1, shape: ['vector:none', 'int'] }
    ]
  }), [
    { regIdx: 0, shape: ['vector:none', 'int'] }
  ]);
  // Tests for both outputs and inputoutputs
  t.deepEqual(registerOpShape({
    outputs: [
      { id: 1, shape: ['1', 'CV_16SC1'] }
    ],
    inputoutputs: [
      { id: 3, shape: ['3', 'float'] }
    ]
  }), [
    { regIdx: 0, shape: ['1', 'CV_16SC1'] },
    { regIdx: 1, shape: ['3', 'float'] }
  ]);
  t.deepEqual(registerOpShape({
    outputs: [
      { id: 3, shape: ['vector:3', 'double'] }
    ],
    inputoutputs: [
      { id: 1, shape: ['vector:1', 'CV_32SC3'] }
    ]
  }), [
    { regIdx: 0, shape: ['vector:1', 'CV_32SC3'] },
    { regIdx: 1, shape: ['vector:3', 'double'] }
  ]);
  t.deepEqual(registerOpShape({
    outputs: [
      { id: 3, shape: ['vector:none', '3', 'char'] },
      { id: 2, shape: ['vector:none', '2', 'CV_8S'] }
    ],
    inputoutputs: [
      { id: 4, shape: ['vector:none', '4', 'CV_64F'] },
      { id: 1, shape: ['vector:none', '1', 'CV_32F'] }
    ]
  }), [
    { regIdx: 0, shape: ['vector:none', '1', 'CV_32F'] },
    { regIdx: 1, shape: ['vector:none', '2', 'CV_8S'] },
    { regIdx: 2, shape: ['vector:none', '3', 'char'] },
    { regIdx: 3, shape: ['vector:none', '4', 'CV_64F'] }
  ]);
});

const SHAPE_INVALID_RANK_MSG = 'Invalid rank number of the shape';

test('registerOpShapeFn: return string for setting output shape', t => {
  // Test for tfRank < 0
  const getTfRankStub = sinon.stub(utils, 'getTfRank');
  getTfRankStub.returns(-1);
  testThrownMsg(t, `${SHAPE_INVALID_RANK_MSG}: -1`, registerOpShapeFn.bind({ regIdx: 0, shape: ['CV_8U'] }));
  getTfRankStub.restore();
  // Tests for tfRank = 0
  t.is(
    registerOpShapeFn.bind({ regIdx: 0, shape: ['CV_8U'] })(),
    'c->set_output(0, c->Scalar());'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 0, shape: ['CV_8UC1'] })(),
    'c->set_output(0, c->Scalar());'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 1, shape: ['int'] })(),
    'c->set_output(1, c->Scalar());'
  );
  // Tests for tfRank = 1
  t.is(
    registerOpShapeFn.bind({ regIdx: 0, shape: ['10', 'CV_8S'] })(),
    'c->set_output(0, c->Vector(10));'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 1, shape: ['none', 'CV_8UC1'] })(),
    'c->set_output(1, c->Vector(InferenceContext::kUnknownDim));'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 2, shape: ['vector:20', 'float'] })(),
    'c->set_output(2, c->Vector(20));'
  );
  // Tests for tfRank = 2
  t.is(
    registerOpShapeFn.bind({ regIdx: 0, shape: ['10', '20', 'CV_16U'] })(),
    'c->set_output(0, c->Matrix(10, 20));'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 1, shape: ['vector:none', '100', 'CV_16UC1'] })(),
    'c->set_output(1, c->Matrix(InferenceContext::kUnknownDim, 100));'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 2, shape: ['vector:30', 'CV_16UC3'] })(),
    'c->set_output(2, c->Matrix(30, 3));'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 3, shape: ['vector:10', 'vector:10', 'char'] })(),
    'c->set_output(3, c->Matrix(10, 10));'
  );
  // Tests for tfRank >= 3
  t.is(
    registerOpShapeFn.bind({ regIdx: 0, shape: ['100', '200', '300', 'CV_32F'] })(),
    'c->set_output(0, c->MakeShape({ 100, 200, 300 }));'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 1, shape: ['vector:100', 'none', '300', 'CV_32FC1'] })(),
    'c->set_output(1, c->MakeShape({ 100, InferenceContext::kUnknownDim, 300 }));'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 2, shape: ['vector:none', 'vector:5', 'CV_32FC3'] })(),
    'c->set_output(2, c->MakeShape({ InferenceContext::kUnknownDim, 5, 3 }));'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 3, shape: ['vector:50', 'vector:none', 'none', 'char'] })(),
    'c->set_output(3, c->MakeShape({ 50, InferenceContext::kUnknownDim, InferenceContext::kUnknownDim }));'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 4, shape: ['vector:none', 'vector:15', 'vector:10', '5', 'CV_64F'] })(),
    'c->set_output(4, c->MakeShape({ InferenceContext::kUnknownDim, 15, 10, 5 }));'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 4, shape: ['vector:none', 'vector:15', 'vector:10', '5', 'CV_64FC2'] })(),
    'c->set_output(4, c->MakeShape({ InferenceContext::kUnknownDim, 15, 10, 5, 2 }));'
  );
});

test('computeInput: convert inputs and inputoutputs of parsed operations metadata to array for preparing inputs', t => {
  const shape = ['3', '3', 'CV_8U'];
  const pShape = { tfRank: 3 };

  // Tests for empty array result
  t.deepEqual(computeInput(), []);
  t.deepEqual(computeInput({}), []);
  t.deepEqual(computeInput({ inputs: [] }), []);
  t.deepEqual(computeInput({ inputoutputs: [] }), []);
  // Test for inputs only
  t.deepEqual(computeInput({
    inputs: [
      { id: 1, name: 'name1', shape, pShape }
    ]
  }), [
    { regIdx: 0, name: 'name1', shape, pShape }
  ]);
  // Test for inputoutputs only
  t.deepEqual(computeInput({
    inputoutputs: [
      { id: 1, name: 'name1', shape, pShape }
    ]
  }), [
    { regIdx: 0, name: 'name1', shape, pShape }
  ]);
  // Tests for both input and inputoutputs
  t.deepEqual(computeInput({
    inputs: [
      { id: 1, name: 'name1', shape, pShape }
    ],
    inputoutputs: [
      { id: 3, name: 'name3', shape, pShape }
    ]
  }), [
    { regIdx: 0, name: 'name1', shape, pShape },
    { regIdx: 1, name: 'name3', shape, pShape }
  ]);
  t.deepEqual(computeInput({
    inputs: [
      { id: 3, name: 'name3', shape, pShape }
    ],
    inputoutputs: [
      { id: 1, name: 'name1', shape, pShape }
    ]
  }), [
    { regIdx: 0, name: 'name1', shape, pShape },
    { regIdx: 1, name: 'name3', shape, pShape }
  ]);
  t.deepEqual(computeInput({
    inputs: [
      { id: 3, name: 'name3', shape, pShape },
      { id: 2, name: 'name2', shape, pShape }
    ],
    inputoutputs: [
      { id: 1, name: 'name1', shape, pShape },
      { id: 4, name: 'name4', shape, pShape }
    ]
  }), [
    { regIdx: 0, name: 'name1', shape, pShape },
    { regIdx: 1, name: 'name2', shape, pShape },
    { regIdx: 2, name: 'name3', shape, pShape },
    { regIdx: 3, name: 'name4', shape, pShape }
  ]);
});

test.skip('computeInputFn: return string of inputs changed from tensorflow to OpenCV', t => {
});

test('computeExecute: convert parsed operations metadata to array for execution', t => {
  const fnName = 'my_func';

  // Test for function name only
  t.deepEqual(computeExecute({ fnName }), { fnName });
  // Test for inputs
  t.deepEqual(computeExecute({
    fnName,
    inputs: [{ id: 0, name: 'input0' }]
  }), {
    fnName,
    inputs: [{ id: 0, name: 'input0' }]
  });
  // Test for outputs
  t.deepEqual(computeExecute({
    fnName,
    outputs: [{ id: 0, name: 'output0' }]
  }), {
    fnName,
    outputs: [{ id: 0, name: 'output0' }]
  });
  // Test for inputoutputs
  t.deepEqual(computeExecute({
    fnName,
    inputoutputs: [{ id: 0, name: 'inputoutput0' }]
  }), {
    fnName,
    inputoutputs: [{ id: 0, name: 'inputoutput0' }]
  });
  // Test for attributes
  t.deepEqual(computeExecute({
    fnName,
    attributes: [{ id: 0, name: 'attr0' }]
  }), {
    fnName,
    attributes: [{ id: 0, name: 'attr0' }]
  });
  // Test for all
  t.deepEqual(computeExecute({
    fnName,
    inputs: [{ id: 0, name: 'input0' }],
    outputs: [{ id: 1, name: 'output1' }],
    inputoutputs: [{ id: 2, name: 'inputoutput2' }],
    attributes: [{ id: 3, name: 'attr3' }]
  }), {
    fnName,
    inputs: [{ id: 0, name: 'input0' }],
    outputs: [{ id: 1, name: 'output1' }],
    inputoutputs: [{ id: 2, name: 'inputoutput2' }],
    attributes: [{ id: 3, name: 'attr3' }]
  });
});

test('computeExecuteFn: return string for execution', t => {
  const fnName = 'my_func';

  // Test for function name only
  t.is(
    computeExecuteFn.bind({ fnName })(),
    `\n  ${fnName}();`
  );
  // Tests for inputs
  t.is(
    computeExecuteFn.bind({
      fnName,
      inputs: []
    })(),
    `\n  ${fnName}();`
  );
  t.is(
    computeExecuteFn.bind({
      fnName,
      inputs: [
        { id: 0, name: 'input0' }
      ]
    })(),
    `\n  ${fnName}(input0_cv);`
  );
  // Tests for outputs
  t.is(
    computeExecuteFn.bind({
      fnName,
      outputs: []
    })(),
    `\n  ${fnName}();`
  );
  t.is(
    computeExecuteFn.bind({
      fnName,
      outputs: [
        { id: 1, name: 'output1' },
        { id: 0, name: 'output0' },
      ]
    })(),
    `Mat output1_cv, output0_cv;\n  ${fnName}(output0_cv, output1_cv);`
  );
  // Tests for inputoutpus
  t.is(
    computeExecuteFn.bind({
      fnName,
      inputoutputs: []
    })(),
    `\n  ${fnName}();`
  );
  t.is(
    computeExecuteFn.bind({
      fnName,
      inputoutputs: [
        { id: 0, name: 'inputoutput0' },
        { id: 1, name: 'inputoutput1' },
      ]
    })(),
    `\n  ${fnName}(inputoutput0_cv, inputoutput1_cv);`
  );
  // Tests for attributes
  t.is(
    computeExecuteFn.bind({
      fnName,
      attributes: []
    })(),
    `\n  ${fnName}();`
  );
  t.is(
    computeExecuteFn.bind({
      fnName,
      attributes: [
        { id: 2, name: 'attr2' },
        { id: 0, name: 'attr0' },
        { id: 1, name: 'attr1' }
      ]
    })(),
    `\n  ${fnName}(attr0_, attr1_, attr2_);`
  );
  // Test for all
  t.is(
    computeExecuteFn.bind({
      fnName,
      inputs: [
        { id: 2, name: 'input2' },
        { id: 4, name: 'input4' }
      ],
      outputs: [
        { id: 6, name: 'output6' },
        { id: 5, name: 'output5' },
        { id: 7, name: 'output7' }
      ],
      inputoutputs: [
        { id: 0, name: 'inputoutput0' },
        { id: 1, name: 'inputoutput1' }
      ],
      attributes: [
        { id: 3, name: 'attr3' },
        { id: 8, name: 'attr8' }
      ]
    })(),
    `Mat output6_cv, output5_cv, output7_cv;\n  ${fnName}(inputoutput0_cv, inputoutput1_cv, input2_cv, attr3_, input4_cv, output5_cv, output6_cv, output7_cv, attr8_);`
  );
});

test('computeOutput: convert outputs and inputoutputs of parsed operations metadata to array for preparing output', t => {
  const shape = ['3', '3', 'CV_8U'];
  const pShape = { tfRank: 3 };

  // Tests for empty array result
  t.deepEqual(computeOutput(), []);
  t.deepEqual(computeOutput({}), []);
  t.deepEqual(computeOutput({ outputs: [] }), []);
  t.deepEqual(computeOutput({ inputoutputs: [] }), []);
  // Test for inputs only
  t.deepEqual(computeOutput({
    outputs: [
      { id: 1, name: 'name1', shape, pShape }
    ]
  }), [
    { regIdx: 0, name: 'name1', shape, pShape }
  ]);
  // Test for inputoutputs only
  t.deepEqual(computeOutput({
    inputoutputs: [
      { id: 1, name: 'name1', shape, pShape }
    ]
  }), [
    { regIdx: 0, name: 'name1', shape, pShape }
  ]);
  // Tests for both input and inputoutputs
  t.deepEqual(computeOutput({
    outputs: [
      { id: 1, name: 'name1', shape, pShape }
    ],
    inputoutputs: [
      { id: 3, name: 'name3', shape, pShape }
    ]
  }), [
    { regIdx: 0, name: 'name1', shape, pShape },
    { regIdx: 1, name: 'name3', shape, pShape }
  ]);
  t.deepEqual(computeOutput({
    outputs: [
      { id: 3, name: 'name3', shape, pShape }
    ],
    inputoutputs: [
      { id: 1, name: 'name1', shape, pShape }
    ]
  }), [
    { regIdx: 0, name: 'name1', shape, pShape },
    { regIdx: 1, name: 'name3', shape, pShape }
  ]);
  t.deepEqual(computeOutput({
    outputs: [
      { id: 3, name: 'name3', shape, pShape },
      { id: 2, name: 'name2', shape, pShape }
    ],
    inputoutputs: [
      { id: 1, name: 'name1', shape, pShape },
      { id: 4, name: 'name4', shape, pShape }
    ]
  }), [
    { regIdx: 0, name: 'name1', shape, pShape },
    { regIdx: 1, name: 'name2', shape, pShape },
    { regIdx: 2, name: 'name3', shape, pShape },
    { regIdx: 3, name: 'name4', shape, pShape }
  ]);
});

test.skip('computeOutputFn: return string of outputs changed from OpenCV to tensorflow', t => {
});

function testIntervalRender(t, opsMeta) {
  const mustacheStub = sinon.stub(Mustache, 'render', () => 'mustache');
  const getTemplateStub = sinon.stub(template, 'get', () => 'template');

  generator.__set__('registerOpInput', () => []);
  generator.__set__('registerOpOutput', () => []);
  generator.__set__('opAttributes', () => []);
  generator.__set__('registerOpShape', () => []);
  generator.__set__('computeInput', () => []);
  generator.__set__('computeExecute', () => ({}));
  generator.__set__('computeOutput', () => []);

  const expectedRenderArg = {
    opName: opsMeta.opName,
    fnName: opsMeta.fnName,
    device: opsMeta.device ? opsMeta.device : 'DEVICE_CPU',
    registerOpInput: generator.__get__('registerOpInput')(),
    registerOpInputFn,
    registerOpOutput: generator.__get__('registerOpOutput')(),
    registerOpOutputFn,
    opAttributes: generator.__get__('opAttributes')(),
    registerOpAttrFn,
    getAttributesFn,
    declareAttributesFn,
    registerOpShape: generator.__get__('registerOpShape')(),
    registerOpShapeFn,
    computeInput: generator.__get__('computeInput')(),
    computeInputFn,
    computeExecute: generator.__get__('computeExecute')(),
    computeExecuteFn,
    computeOutput: generator.__get__('computeOutput')(),
    computeOutputFn
  };

  // TODO: Check arguments that is passed to registar functions
  t.is(_render(opsMeta), mustacheStub());
  t.true(mustacheStub.calledWith(getTemplateStub(), expectedRenderArg));

  mustacheStub.restore();
  getTemplateStub.restore();
  generator.__set__('registerOpInput', registerOpInput);
  generator.__set__('registerOpOutput', registerOpOutput);
  generator.__set__('opAttributes', opAttributes);
  generator.__set__('registerOpShape', registerOpShape);
  generator.__set__('computeInput', computeInput);
  generator.__set__('computeExecute', computeExecute);
  generator.__set__('computeOutput', computeOutput);
}

test('_render: internal function for render', t => {
  const opName = 'my_op';
  const fnName = 'my_fn';
  const inputs = {
    input0: { id: 0, shape: ['CV_8U'] },
    input1: { id: 1, shape: ['none', 'int'] }
  };
  const outputs = {
    output2: { id: 2, shape: ['10', 'float'] },
    output3: { id: 3, shape: ['20', 'none', 'CV_16U'] },
    output4: { id: 4, shape: ['CV_32S'] },
  };
  const inputoutputs = {
    inputoutput5: { id: 5, shape: ['none', 'none', 'CV_64FC3'] },
  };
  const attributes = {
    attr6: { id: 6, type: 'int = 30' },
    attr7: { id: 7, type: 'string = "this is string"' }
  };

  testIntervalRender(t, {
    opName,
    fnName,
    inputs,
    outputs,
    inputoutputs,
    attributes
  });
});

test.skip('render: external function for rendering output file', t => {
});
