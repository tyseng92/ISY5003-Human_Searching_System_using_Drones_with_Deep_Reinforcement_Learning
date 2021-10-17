Update Fix
====================================================================
17 Oct 2021
- Successfully solve the Path Follow problem.
- Done changing the animation posture for the idle standing pose.
====================================================================

Problem encountered:
- the human character does not follow the PatrolPath set in the environment after choosing the 'AirSimGameMode' in the World Settings tab. 

Cause:
- might be due to only c++ code is being used instead of Blueprint(visual scripting for UE4). Need to run Blueprint from Airsim package to activate the other Blueprint for NPC (Content/AI/NPC).

To fix the Path Follow problem for human character in UE4:
- try to include the blueprint (BP_Flying_Pawn) from AirSim Content in Content Browser. Doing this allow the blueprint for NPC to be activated and execute the Path Follow using the PatrolPath waypoints for the human character. 
- remember to set the 'Add Possess Player' to 'Player 0' in Pawn section in detail tab.

Problem:
- Unable to run the .exe file due to the 'Array index out of bound' error (https://answers.unrealengine.com/questions/835900/weird-array-index-out-of-bounds-error-causing-cras.html?sort=oldest)

Useful links:
- https://www.youtube.com/watch?v=1oY8Qu5maQQ (airsim custom environment)

- https://www.youtube.com/watch?v=zNJEvAGiw7w&list=PL4G2bSPE_8ukuajpXPlAE47Yez7EAyKMu (ryan laley ai)

- https://www.youtube.com/watch?v=Io76DagpS-8 (retargeting new mesh to skeleton)

- https://www.mixamo.com/#/ (mixamo for more animation)

- https://www.unrealengine.com/marketplace/en-US/product/bafae2f71777417c8864a4cca9f47f2d (assets)

- https://www.unrealengine.com/marketplace/en-US/product/9c3fab270dfe468a9a920da0c10fa2ad (scanned 3d people pack)