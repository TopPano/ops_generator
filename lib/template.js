const template = `
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

module.exports = {
  get: () => {
    return template;
  }
};
