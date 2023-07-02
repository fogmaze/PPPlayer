#include "PxConfig.h"
#include "PxPhysicsAPI.h"
#include "vehicle/PxVehicleSDK.h"
#include "extensions/PxExtensionsAPI.h"
#include "extensions/PxDefaultErrorCallback.h"
#include "extensions/PxDefaultAllocator.h"

using namespace physx;

void simulate() {
    PxDefaultErrorCallback gDefaultErrorCallback;
    PxDefaultAllocator gDefaultAllocatorCallback;
    PxSimulationFilterShader gDefaultFilterShader = PxDefaultSimulationFilterShader;
    PxFoundation* gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gDefaultAllocatorCallback, gDefaultErrorCallback);
    PxPhysics* gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true);
    PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    PxDefaultCpuDispatcher* gDispatcher = PxDefaultCpuDispatcherCreate(2);
    sceneDesc.cpuDispatcher = gDispatcher;
    sceneDesc.filterShader = gDefaultFilterShader;
    PxScene* gScene = gPhysics->createScene(sceneDesc);
    PxMaterial* gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);
    PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
    gScene->addActor(*groundPlane);
    gScene->simulate(1.0f / 60.0f);
}

int main() {
    simulate();
}