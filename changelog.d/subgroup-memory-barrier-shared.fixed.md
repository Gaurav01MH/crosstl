GLSL output now refuses `subgroupMemoryBarrierShared()` outside compute, mesh and task stages with a diagnostic instead of emitting invalid fragment-stage code (#5027).
