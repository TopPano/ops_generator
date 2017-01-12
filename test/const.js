module.exports = {
  // List of OpenCV depths, please ref to:
  // http://ninghang.blogspot.tw/2012/11/list-of-mat-type-in-opencv.html or
  // http://www.prism.gatech.edu/~ahuaman3/docs/OpenCV_Docs/tutorials/basic_0/basic_0.html
  VALID_CV_DEPTHS: ['8U', '8S', '16U', '16S', '32S', '32F', '64F'],
  INVALID_CV_DEPTHS: ['8F', '16F', '32U', '64U', '64S'],
  VALID_STD_TYPES: ['char', 'int', 'float', 'double'],
  // Maximum of OpenCV channels:
  // http://docs.opencv.org/2.4/modules/core/doc/intro.html#fixed-pixel-types-limited-use-of-templates
  MAX_CV_CHANNELS: 4
};
