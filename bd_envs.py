"""
bd_envs.py will include all our environmental models that are separate from 
the core models that this ABM needs to run
"""


from model import IntermediateLevelEnv, LowestLevelEnv

class DivisionEnv(IntermediateLevelEnv):
    pass

class ZillaEnv(IntermediateLevelEnv):
    pass

class UpazillaEnv(IntermediateLevelEnv):
    pass

class UnionEnv(IntermediateLevelEnv):
    pass

class MahallaEnv(LowestLevelEnv):
    pass

class VillageEnv(LowestLevelEnv):
    pass