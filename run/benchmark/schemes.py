from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class JointScheme:
    name: str
    use_ranking: bool
    use_caching: bool
    v2i_only: bool
    provider_scope: str = "all"

    def context_flags(self) -> Dict[str, Any]:
        return {
            "use_ranking": self.use_ranking,
            "use_caching": self.use_caching,
            "v2i_only": self.v2i_only,
            "provider_scope": self.provider_scope,
        }


SCHEMES: Dict[str, JointScheme] = {
    "dcsga": JointScheme("dcsga", True, True, False, "all"),
    "dtosc": JointScheme("dtosc", True, True, False, "local_and_rsu"),
    "to_v2i": JointScheme("to_v2i", True, True, True, "rsu_only"),
    "to_wo_c": JointScheme("to_wo_c", True, False, False, "all"),
    "to_wo_r": JointScheme("to_wo_r", False, True, False, "all"),
    "gwo_aco": JointScheme("gwo_aco", True, True, False, "all"),
    "gwo": JointScheme("gwo", True, True, False, "all"),
    "gpc": JointScheme("gpc", True, True, False, "all"),
    "cpo": JointScheme("cpo", True, True, False, "all"),
    "dcpo_base": JointScheme("dcpo_base", True, True, False, "all"),
    "dcpo_criticality": JointScheme("dcpo_criticality", True, True, False, "all"),
    "dcpo_cache": JointScheme("dcpo_cache", True, True, False, "all"),
    "puma": JointScheme("puma", True, True, False, "all"),
}


SUPPORTED_JOINT_ALGORITHMS = tuple(SCHEMES)


def get_joint_scheme(name: str) -> JointScheme:
    key = str(name).strip().lower()

    if key not in SCHEMES:
        raise ValueError(f"Unsupported joint algorithm: {name}")

    return SCHEMES[key]
