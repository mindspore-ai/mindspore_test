#include <vector>
#include "acl/acl_base.h"

typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;
typedef struct aclScalarList aclScalarList;

extern "C" int aclnnDecoderKvCacheGetWorkSpaceSize(void *func_ptr, std::vector<void *> inputs, std::vector<void *> outputs,
                                              uint64_t *workspace_size, aclOpExecutor **executor) {
  using FuncType = int (*)(aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *,
                           aclTensor *, uint64_t *, aclOpExecutor **);
  auto func = reinterpret_cast<FuncType>(func_ptr);
  aclTensor *input0 = static_cast<aclTensor *>(inputs[0]);
  aclTensor *input1 = static_cast<aclTensor *>(inputs[1]);
  aclTensor *input2 = static_cast<aclTensor *>(inputs[2]);
  aclTensor *input3 = static_cast<aclTensor *>(inputs[3]);
  aclTensor *input4 = static_cast<aclTensor *>(inputs[4]);
  aclTensor *input5 = static_cast<aclTensor *>(inputs[5]);
  aclTensor *input6 = static_cast<aclTensor *>(inputs[6]);
  aclTensor *output0 = static_cast<aclTensor *>(outputs[0]);
  return func(input0, input1, input2, input3, input4, input5, input6, output0, workspace_size, executor);
}
