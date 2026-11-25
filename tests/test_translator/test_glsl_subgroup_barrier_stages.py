"""GLSL subgroup barriers are refused in stages where the extension does not define them."""

import pytest

import crosstl
from crosstl.translator.semantics.builtin_values import BuiltinValueUnsupportedError


FRAGMENT = """shader Probe {
    fragment {
        vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {
            BARRIER();
            return vec4(pos.x);
        }
    }
}
"""

COMPUTE = """shader Probe {
    compute {
        @workgroup_size(1, 1, 1)
        void main() {
            subgroupMemoryBarrierShared();
        }
    }
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="opengl", format_output=False)


def test_shared_subgroup_barrier_is_refused_in_fragment_stage(tmp_path):
    with pytest.raises(
        BuiltinValueUnsupportedError,
        match="subgroupMemoryBarrierShared.*fragment stage",
    ):
        _translate(tmp_path, FRAGMENT.replace("BARRIER", "subgroupMemoryBarrierShared"))


def test_shared_subgroup_barrier_is_allowed_in_compute_stage(tmp_path):
    assert _translate(tmp_path, COMPUTE)


@pytest.mark.parametrize(
    "barrier",
    [
        "subgroupBarrier",
        "subgroupMemoryBarrier",
        "subgroupMemoryBarrierBuffer",
        "subgroupMemoryBarrierImage",
        "subgroupExecutionBarrier",
    ],
)
def test_other_subgroup_barriers_remain_available_in_fragment_stage(tmp_path, barrier):
    assert _translate(tmp_path, FRAGMENT.replace("BARRIER", barrier))
