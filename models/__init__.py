from .reskanet import reskanet_18x32p, reskacnet_18x32p, fast_reskanet_18x32p, reskalnet_18x32p, reskalnet_18x64p, \
    reskalnet_50x64p, moe_reskalnet_50x64p, reskalnet_101x64p, moe_reskalnet_101x64p, \
    reskalnet_152x64p, moe_reskalnet_152x64p, moe_reskalnet_18x64p, reskalnet_101x32p, \
    reskalnet_152x32p, reskagnet_101x64p, reskagnet_18x32p
from .densekanet import densekanet121, densekalnet121, densekacnet121, densekagnet121, fast_densekanet121
from .densekanet import densekalnet161, densekalnet169, densekalnet201
from .densekanet import tiny_densekanet, tiny_densekalnet, tiny_densekacnet, tiny_fast_densekanet, tiny_densekagnet
from .ukanet import ukanet_18, ukalnet_18, fast_ukanet_18, ukacnet_18, ukagnet_18
from .vggkan import fast_vggkan, vggkan, vggkaln, vggkacn, vggkagn
from .lekanet import LeKANet, Fast_LeKANet, LeKALNet, LeKACNet, LeKAGNet

from .baselines import LeNet