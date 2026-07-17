/**
 * Production environment configuration.
 * Uses P5e (H200) instances for full-scale RL training.
 * Higher availability for OSMO control plane.
 */
export const prodConfig = {
  environment: 'prod',

  // EKS Cluster
  eks: {
    clusterName: 'physical-ai-prod',
    version: '1.31',
    controlPlaneNodes: {
      instanceType: 'm6i.xlarge',
      minSize: 3,
      maxSize: 5,
      desiredSize: 3,
    },
    gpuNodes: {
      instanceType: 'p5e.48xlarge', // H200 — full training power
      minSize: 0,
      maxSize: 4,
      desiredSize: 0, // Scale from zero, OSMO triggers scale-up
      diskSize: 500, // GB — large datasets + containers
    },
  },

  // OSMO Dependencies
  osmo: {
    postgres: {
      instanceType: 'db.r6g.large',
      allocatedStorage: 100, // GB
      multiAz: true,
    },
    redis: {
      nodeType: 'cache.r6g.large',
      numNodes: 2,
    },
  },

  // Storage
  storage: {
    bucketPrefix: 'physical-ai-prod',
  },

  // Edge / IoT
  edge: {
    thingGroupName: 'physical-ai-prod-robots',
  },
};
