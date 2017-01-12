import test from 'ava';
import types from '../lib/types';

// 1. List of CvTypes (without CV_ prefix and channel postfix): 8U, 8S, 16U, 16S, 32S, 32F, 64F
//    For details, please ref to:
//    http://ninghang.blogspot.tw/2012/11/list-of-mat-type-in-opencv.html or
//    http://www.prism.gatech.edu/~ahuaman3/docs/OpenCV_Docs/tutorials/basic_0/basic_0.html
// 2. List of tensorflow data types: https://www.tensorflow.org/versions/master/resources/dims_types#data-types

const cvErrMsg = 'Invalid data type format of CV';
const stdErrMsg = 'Type not supported';

function checkErrMsg(t, fn, errMsg, type) {
  t.throws(() => fn(type), `${errMsg}: ${type}`);
}

test('cvToStd: translate a CvType (without CV_ prefix and channel postfix) to standard C type', t => {
  checkErrMsg(t, types.cvToStd, cvErrMsg);
  t.is(types.cvToStd('8U'), 'uint8_t');
  t.is(types.cvToStd('8S'), 'int8_t');
  checkErrMsg(t, types.cvToStd, cvErrMsg, '8F');
  t.is(types.cvToStd('16U'), 'uint16_t');
  t.is(types.cvToStd('16S'), 'int16_t');
  checkErrMsg(t, types.cvToStd, cvErrMsg, '16F');
  checkErrMsg(t, types.cvToStd, cvErrMsg, '32U');
  t.is(types.cvToStd('32S'), 'int32_t');
  t.is(types.cvToStd('32F'), 'float');
  checkErrMsg(t, types.cvToStd, cvErrMsg, '64U');
  checkErrMsg(t, types.cvToStd, cvErrMsg, '64S');
  t.is(types.cvToStd('64F'), 'double');
});

test('cvToTf: translate a CvType (without CV_ prefix and channel postfix) to tensorflow type', t => {
  checkErrMsg(t, types.cvToTf, cvErrMsg);
  t.is(types.cvToTf('8U'), 'uint8');
  t.is(types.cvToTf('8S'), 'int8');
  checkErrMsg(t, types.cvToTf, cvErrMsg, '8F');
  t.is(types.cvToTf('16U'), 'uint16');
  t.is(types.cvToTf('16S'), 'int16');
  checkErrMsg(t, types.cvToTf, cvErrMsg, '16F');
  checkErrMsg(t, types.cvToTf, cvErrMsg, '32U');
  t.is(types.cvToTf('32S'), 'int32');
  t.is(types.cvToTf('32F'), 'float32');
  checkErrMsg(t, types.cvToTf, cvErrMsg, '64U');
  checkErrMsg(t, types.cvToTf, cvErrMsg, '64S');
  t.is(types.cvToTf('64F'), 'float64');
});

test('stdToCv: translate a standard C type to CvType (without channel postfix)', t => {
  checkErrMsg(t, types.stdToCv, stdErrMsg);
  t.is(types.stdToCv('char'), 'CV_8S');
  t.is(types.stdToCv('int'), 'CV_32S');
  t.is(types.stdToCv('float'), 'CV_32F');
  t.is(types.stdToCv('double'), 'CV_64F');
});

test('stdToCv: translate a standard C type to tensorflow type', t => {
  checkErrMsg(t, types.stdToTf, stdErrMsg);
  t.is(types.stdToTf('char'), 'int8');
  t.is(types.stdToTf('int'), 'int32');
  t.is(types.stdToTf('float'), 'float32');
  t.is(types.stdToTf('double'), 'float64');
});
