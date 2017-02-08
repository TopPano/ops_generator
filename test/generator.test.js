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
import { parseShape } from '../lib/shape';

const generator = rewire('../lib/generator');
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
const renderKernelTemp = generator.__get__('renderKernelTemp');


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
      { id: 1, name: 'input1', typeFormat: 'std', dtype: 'float' }
    ]
  }), [
    { name: 'input1', type: 'float32' }
  ]);
  // Test for inputoutputs only
  t.deepEqual(registerOpInput({
    inputoutputs: [
      { id: 1, name: 'inputoutput1', typeFormat: 'cv', dtype: '8S' }
    ]
  }), [
    { name: 'inputoutput1_in', type: 'int8' }
  ]);
  // Tests for inputs and inputoutputs
  t.deepEqual(registerOpInput({
    inputs: [
      { id: 1, name: 'input1', typeFormat: 'std', dtype: 'float' }
    ],
    inputoutputs: [
      { id: 3, name: 'inputoutput3', typeFormat: 'cv', dtype: '8S' }
    ]
  }), [
    { name: 'input1', type: 'float32' },
    { name: 'inputoutput3_in', type: 'int8' }
  ]);
  t.deepEqual(registerOpInput({
    inputs: [
      { id: 3, name: 'input3', typeFormat: 'cv', dtype: '8S' }
    ],
    inputoutputs: [
      { id: 1, name: 'inputoutput1', typeFormat: 'std', dtype: 'float' }
    ]
  }), [
    { name: 'inputoutput1_in', type: 'float32' },
    { name: 'input3', type: 'int8' }
  ]);
  t.deepEqual(registerOpInput({
    inputs: [
      { id: 3, name: 'input3', typeFormat: 'cv', dtype: '8S' },
      { id: 2, name: 'input2', typeFormat: 'std', dtype: 'int' }
    ],
    inputoutputs: [
      { id: 4, name: 'inputoutput4', typeFormat: 'std', dtype: 'double' },
      { id: 1, name: 'inputoutput1', typeFormat: 'cv', dtype: '32F' }
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
      { id: 1, name: 'output1', typeFormat: 'std', dtype: 'float' }
    ]
  }), [
    { name: 'output1', type: 'float32' }
  ]);

  // Test for inputoutputs only
  t.deepEqual(registerOpOutput({
    inputoutputs: [
      { id: 1, name: 'inputoutput1', typeFormat: 'cv', dtype: '8S' }
    ]
  }), [
    { name: 'inputoutput1', type: 'int8' }
  ]);

  // Tests for both output and inputoutputs
  t.deepEqual(registerOpOutput({
    outputs: [
      { id: 1, name: 'output1', typeFormat: 'std', dtype: 'float' }
    ],
    inputoutputs: [
      { id: 3, name: 'inputoutput3', typeFormat: 'cv', dtype: '8S' }
    ]
  }), [
    { name: 'output1', type: 'float32' },
    { name: 'inputoutput3', type: 'int8' }
  ]);
  t.deepEqual(registerOpOutput({
    outputs: [
      { id: 3, name: 'output3', typeFormat: 'cv', dtype: '8S' }
    ],
    inputoutputs: [
      { id: 1, name: 'inputoutput1', typeFormat: 'std', dtype: 'float' }
    ]
  }), [
    { name: 'inputoutput1', type: 'float32' },
    { name: 'output3', type: 'int8' }
  ]);
  t.deepEqual(registerOpOutput({
    outputs: [
      { id: 3, name: 'output3', typeFormat: 'cv', dtype: '8S' },
      { id: 2, name: 'output2', typeFormat: 'std', dtype: 'int' }
    ],
    inputoutputs: [
      { id: 4, name: 'inputoutput4', typeFormat: 'std', dtype: 'double' },
      { id: 1, name: 'inputoutput1', typeFormat: 'cv', dtype: '32F' }
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
      { id: 1, pShape: 'pShapeMock' }
    ]
  }), [
    { regIdx: 0, pShape: 'pShapeMock' }
  ]);
  // Test for inputoutputs only
  t.deepEqual(registerOpShape({
    inputoutputs: [
      { id: 1, pShape: 'pShapeMock' }
    ]
  }), [
    { regIdx: 0, pShape: 'pShapeMock' }
  ]);
  // Tests for both outputs and inputoutputs
  t.deepEqual(registerOpShape({
    outputs: [
      { id: 1, pShape: 'pShapeMock1' }
    ],
    inputoutputs: [
      { id: 3, pShape: 'pShapeMock2' }
    ]
  }), [
    { regIdx: 0, pShape: 'pShapeMock1' },
    { regIdx: 1, pShape: 'pShapeMock2' }
  ]);
  t.deepEqual(registerOpShape({
    outputs: [
      { id: 3, pShape: 'pShapeMock1' }
    ],
    inputoutputs: [
      { id: 1, pShape: 'pShapeMock2' }
    ]
  }), [
    { regIdx: 0, pShape: 'pShapeMock2' },
    { regIdx: 1, pShape: 'pShapeMock1' }
  ]);
  t.deepEqual(registerOpShape({
    outputs: [
      { id: 3, pShape: 'pShapeMock1' },
      { id: 2, pShape: 'pShapeMock2' }
    ],
    inputoutputs: [
      { id: 4, pShape: 'pShapeMock3' },
      { id: 1, pShape: 'pShapeMock4' }
    ]
  }), [
    { regIdx: 0, pShape: 'pShapeMock4' },
    { regIdx: 1, pShape: 'pShapeMock2' },
    { regIdx: 2, pShape: 'pShapeMock1' },
    { regIdx: 3, pShape: 'pShapeMock3' }
  ]);
});

const SHAPE_INVALID_RANK_MSG = 'Invalid rank number of the shape';

test('registerOpShapeFn: return string for setting output shape', t => {
  // Tests for tfRank = 0
  t.is(
    registerOpShapeFn.bind({ regIdx: 1, pShape: parseShape(['int']) })(),
    'c->set_output(1, c->Scalar());'
  );
  // Tests for tfRank = 1
  t.is(
    registerOpShapeFn.bind({ regIdx: 0, pShape: parseShape(['10', 'CV_8S']) })(),
    'c->set_output(0, c->Vector(10));'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 1, pShape: parseShape(['none', 'CV_8UC1']) })(),
    'c->set_output(1, c->Vector(InferenceContext::kUnknownDim));'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 2, pShape: parseShape(['vector:20', 'float']) })(),
    'c->set_output(2, c->Vector(20));'
  );
  // Tests for tfRank = 2
  t.is(
    registerOpShapeFn.bind({ regIdx: 0, pShape: parseShape(['10', '20', 'CV_16U']) })(),
    'c->set_output(0, c->Matrix(10, 20));'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 1, pShape: parseShape(['vector:none', '100', 'CV_16UC1']) })(),
    'c->set_output(1, c->Matrix(InferenceContext::kUnknownDim, 100));'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 3, pShape: parseShape(['vector:10', 'vector:10', 'char']) })(),
    'c->set_output(3, c->Matrix(10, 10));'
  );
  // Tests for tfRank >= 3
  t.is(
    registerOpShapeFn.bind({ regIdx: 0, pShape: parseShape(['100', '200', '300', 'CV_32F']) })(),
    'c->set_output(0, c->MakeShape({ 100, 200, 300 }));'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 1, pShape: parseShape(['vector:100', 'none', '300', 'CV_32FC1']) })(),
    'c->set_output(1, c->MakeShape({ 100, InferenceContext::kUnknownDim, 300 }));'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 4, pShape: parseShape(['vector:none', 'vector:15', 'vector:10', '5', 'CV_64F']) })(),
    'c->set_output(4, c->MakeShape({ InferenceContext::kUnknownDim, 15, 10, 5 }));'
  );
  t.is(
    registerOpShapeFn.bind({ regIdx: 4, pShape: parseShape(['vector:none', 'vector:15', 'vector:10', '5', 'CV_64FC2']) })(),
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
  const pShape = { tfRank: 3 };

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
    outputs: [{ id: 0, name: 'output0', pShape }]
  }), {
    fnName,
    outputs: [{ id: 0, name: 'output0', pShape }]
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
    outputs: [{ id: 1, name: 'output1', pShape }],
    inputoutputs: [{ id: 2, name: 'inputoutput2' }],
    attributes: [{ id: 3, name: 'attr3' }]
  }), {
    fnName,
    inputs: [{ id: 0, name: 'input0' }],
    outputs: [{ id: 1, name: 'output1', pShape }],
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
        { id: 1, name: 'output1', pShape: { varDecStr: 'int' } },
        { id: 0, name: 'output0', pShape: { varDecStr: 'Mat' } },
        { id: 3, name: 'output3', pShape: { varDecStr: 'vector<vector<double>>' } },
        { id: 2, name: 'output2', pShape: { varDecStr: 'vector<Mat>' } },
      ]
    })(),
    `int output1_cv;\nMat output0_cv;\nvector<vector<double>> output3_cv;\nvector<Mat> output2_cv;\n\n  ${fnName}(output0_cv, output1_cv, output2_cv, output3_cv);`
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
        { id: 6, name: 'output6', pShape: { varDecStr: 'vector<double>' } },
        { id: 5, name: 'output5', pShape: { varDecStr: 'Mat' } },
        { id: 7, name: 'output7', pShape: { varDecStr: 'char' } }
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
    `vector<double> output6_cv;\nMat output5_cv;\nchar output7_cv;\n\n  ${fnName}(inputoutput0_cv, inputoutput1_cv, input2_cv, attr3_, input4_cv, output5_cv, output6_cv, output7_cv, attr8_);`
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
  t.is(renderKernelTemp(opsMeta), mustacheStub());
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

test.skip('renderKernelTemp: internal function for render', t => {
  const opName = 'my_op';
  const fnName = 'my_fn';
  const inputs = {
    input0: { id: 0, shape: ['none', 'CV_8U'] },
    input1: { id: 1, shape: ['int'] }
  };
  const outputs = {
    output2: { id: 2, shape: ['10', 'CV_32F'] },
    output3: { id: 3, shape: ['20', 'none', 'CV_16U'] },
    output4: { id: 4, shape: ['none', 'CV_32S'] },
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
