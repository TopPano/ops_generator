const kernelTemplate = `
/** This file was generated automatically, please don't modify it unless you know what you are doing. **/

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "opencv2/core.hpp"

#include "{{srcFile}}.hpp"

using namespace std;
using namespace pv;

namespace tensorflow {

using shape_inference::InferenceContext;

REGISTER_OP("{{opName}}")
  {{#registerOpInput}}
  {{{registerOpInputFn}}}
  {{/registerOpInput}}
  {{#opAttributes}}
  {{{registerOpAttrFn}}}
  {{/opAttributes}}
  {{#registerOpOutput}}
  {{{registerOpOutputFn}}}
  {{/registerOpOutput}}
  .SetShapeFn([](InferenceContext* c) {
    {{#registerOpShape}}
    {{{registerOpShapeFn}}}
    {{/registerOpShape}}
    return Status::OK();
  });

class {{opName}}Op : public OpKernel {
  public:
    explicit {{opName}}Op(OpKernelConstruction* context) : OpKernel(context) {
      {{#opAttributes}}
      {{{getAttributesFn}}}
      {{/opAttributes}}
    }

    void Compute(OpKernelContext* context) override {
      {{#computeInput}}
      {{{computeInputFn}}}
      {{/computeInput}}

      {{#computeExecute}}
      {{{computeExecuteFn}}}
      {{/computeExecute}}

      {{#computeOutput}}
      {{{computeOutputFn}}}
      {{/computeOutput}}
    }

  private:
    {{#opAttributes}}
    {{{declareAttributesFn}}}
    {{/opAttributes}}
};

REGISTER_KERNEL_BUILDER(Name("{{opName}}").Device({{device}}), {{opName}}Op);

} // namespace tensorflow`;

const pyWrapperTemplate = `
# This file was generated automatically, please don't modify it unless you know what you are doing.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

_op_module = tf.load_op_library(os.path.join(
  tf.resource_loader.get_data_files_path(), '{{kernelSharedLib}}'
))

{{#ops}}
{{name}} = _op_module.{{name}}
{{/ops}}`;

module.exports = {
  getKernelTemp: () => {
    return kernelTemplate;
  },
  getPyWrapperTemp: () => {
    return pyWrapperTemplate;
  }
};
