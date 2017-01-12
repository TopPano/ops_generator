'use strict';

function cvToStd(typeStr) {
  switch (typeStr) {
      case '8U':
          return 'uint8_t';
      case '16U':
          return 'uint16_t';
      case '8S':
          return 'int8_t';
      case '16S':
          return 'int16_t';
      case '32S':
          return 'int32_t';
      case '32F':
          return 'float';
      case '64F':
          return 'double';
      default:
          throw new Error(`Invalid data type format of CV: ${typeStr}`);
  }
}

function cvToTf(typeStr) {
  // XXX: Never find uint32 or uint64 been used in Tensorflow source code...
  switch (typeStr) {
      case '8U':
          return 'uint8';
      case '16U':
          return 'uint16';
      case '8S':
          return 'int8';
      case '16S':
          return 'int16';
      case '32S':
          return 'int32';
      case '32F':
          return 'float32';
      case '64F':
          return 'float64';
      default:
          throw new Error(`Invalid data type format of CV: ${typeStr}`);
  }
}

function stdToCv(typeStr) {
  switch (typeStr) {
      case 'char':
          return 'CV_8S';
      case 'int':
          return 'CV_32S';
      case 'float':
          return 'CV_32F';
      case 'double':
          return 'CV_64F';
      default:
          throw new Error(`Type not supported: ${typeStr}`);
  }
}

function stdToTf(typeStr) {
  switch (typeStr) {
      case 'char':
          return 'int8';
      case 'int':
          return 'int32';
      case 'float':
          return 'float32';
      case 'double':
          return 'float64';
      default:
          throw new Error(`Type not supported: ${typeStr}`);
  }
}

module.exports = {
  cvToStd,
  stdToCv,
  cvToTf,
  stdToTf
};
