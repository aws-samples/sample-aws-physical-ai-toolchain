/**
 * Development environment configuration.
 * Uses smaller/cheaper GPU instances (g5.xlarge) for testing the pipeline
 * without burning through P5e costs.
 */
export const devConfig = {
  environment: 'dev',

  // EKS Cluster
  eks: {
    clusterName: 'physical-ai-dev',
    version: '1.31',
    controlPlaneNodes: {
      instanceType: 'm6i.large',
      minSize: 2,
      maxSize: 3,
      desiredSize: 2,
    },
    gpuNodes: {
      instanceType: 'g5.xlarge', // A10G GPU — cheaper for dev/testing
      minSize: 0,
      maxSize: 2,
      desiredSize: 1,
      diskSize: 200, // GB — Isaac Sim containers are large
    },
  },

  // OSMO Dependencies
  osmo: {
    postgres: {
      instanceType: 'db.t4g.medium',
      allocatedStorage: 20, // GB
      multiAz: false,
    },
    redis: {
      nodeType: 'cache.t4g.small',
      numNodes: 1,
    },
  },

  // Storage
  storage: {
    bucketPrefix: 'physical-ai-dev',
  },

  // Edge / IoT
  edge: {
    thingGroupName: 'physical-ai-dev-robots',
  },
};
