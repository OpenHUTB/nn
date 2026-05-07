/*!
**************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
* Modified from
*https://github.com/OpenGVLab/InternImage
**************************************************************************************************
*/

#include "dscn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dscn_forward", &dscn_forward, "dscn_forward");
    m.def("dscn_backward", &dscn_backward, "dscn_backward");
}
