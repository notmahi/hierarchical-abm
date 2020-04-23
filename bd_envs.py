"""
bd_envs.py will include all our environmental models that are separate from 
the core models that this ABM needs to run
"""


from model import IntermediateLevelEnv, LowestLevelEnv

class ZillaEnv(IntermediateLevelEnv):
    def own_step(self):
        """
        Each subclass needs to define what happens in the own_step. This step is
        where the 2-d spatial simulation happens and people are "infected".
        """
        pass

class UpazillaEnv(IntermediateLevelEnv):
    pass

class UnionEnv(IntermediateLevelEnv):
    pass

class MahallaEnv(LowestLevelEnv):
    pass

class VillageEnv(LowestLevelEnv):
    pass