import * as cdk from 'aws-cdk-lib';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as codebuild from 'aws-cdk-lib/aws-codebuild';
import * as s3assets from 'aws-cdk-lib/aws-s3-assets';
import * as cr from 'aws-cdk-lib/custom-resources';
import { Construct } from 'constructs';
import * as path from 'path';
import { createHash } from 'crypto';

export interface ContainerBuildProps {
  /**
   * ECR repository the built image is pushed to. The image is tagged `:latest`.
   */
  readonly repository: ecr.Repository;

  /**
   * Path to the buildspec file, relative to the repo root.
   * e.g. `containers/gr00t-training/buildspec.yml`
   */
  readonly buildSpecPath: string;

  /**
   * CodeBuild compute size. Large images (Isaac Lab ~16 GB, Cosmos ~30 GB)
   * need X2_LARGE for the extra memory and disk. Default: LARGE.
   */
  readonly computeType?: codebuild.ComputeType;

  /**
   * Build timeout. Pulling a 15 GB NGC base + building can take a while.
   * Default: 1 hour.
   */
  readonly timeout?: cdk.Duration;

  /**
   * Extra environment variables exposed to the buildspec, on top of the
   * standard ECR_REPO_URI / AWS_ACCOUNT_ID / AWS_DEFAULT_REGION set here.
   */
  readonly environmentVariables?: Record<string, codebuild.BuildEnvironmentVariable>;

  /**
   * When true, the build logs in to nvcr.io using the NGC API key from
   * Secrets Manager (secret name `${projectName}/ngc-api-key`). Required for
   * any image whose base comes from NGC (Isaac Lab, Isaac Sim, Cosmos).
   */
  readonly requiresNgcLogin?: boolean;

  /**
   * When true, grants the build IAM permission to PULL from the AWS Deep
   * Learning Container ECR account (763104351884). Required for any image whose
   * base is an AWS DLC (e.g. groot-training's pytorch-inference base) — without
   * it, `docker build` 403s resolving the base manifest even though the
   * buildspec logged in. The buildspec must still `docker login` to that account.
   */
  readonly requiresDlcLogin?: boolean;

  /**
   * Project name prefix, used to name the CodeBuild project and locate the
   * NGC secret. e.g. `physical-ai`.
   */
  readonly projectName: string;

  /**
   * Short, unique name for this image build (becomes part of the CodeBuild
   * project name). e.g. `groot-training`.
   */
  readonly imageName: string;

  /**
   * The shared S3 asset packaging the repo source. Created once by the
   * stack and reused across every ContainerBuild so the (large) source is
   * uploaded only once per deploy.
   */
  readonly sourceAsset: s3assets.Asset;

  /**
   * Auto-trigger a build on every `cdk deploy` where the source changed.
   * Default: true. When false, the project is created but only runs when
   * the user calls `aws codebuild start-build`.
   */
  readonly autoTrigger?: boolean;

  /**
   * Builder CPU architecture. Default 'x86_64' (most images). Use 'arm64' to
   * build aarch64 images natively on a Graviton CodeBuild fleet — e.g. the
   * Jetson (L4T/aarch64) inference image, which can't be built on x86. This
   * keeps even ARM targets off the developer's laptop.
   */
  readonly architecture?: 'x86_64' | 'arm64';
}

/**
 * ContainerBuild — builds a container image in AWS CodeBuild and pushes it to
 * ECR, so users never pull multi-GB NVIDIA base images or build locally.
 *
 * Flow (the standard AWS "no local Docker" pattern):
 *   1. The repo source is packaged into a CDK S3 Asset (uploaded once, shared).
 *   2. A CodeBuild project consumes that S3 source and runs the image's
 *      buildspec on a privileged builder — x86_64 by default, or a Graviton
 *      ARM fleet (architecture: 'arm64') for aarch64/Jetson images. Either way
 *      nothing builds on the developer's laptop.
 *   3. The buildspec logs in to ECR (and optionally NGC), builds, and pushes.
 *   4. An AwsCustomResource fires `startBuild` on create/update. Its physical
 *      ID embeds the asset hash, so builds re-run only when source changes.
 *
 * WORKSHOP NOTE: This is why the labs say "wait for CodeBuild" instead of
 * "docker build". The heavy lifting happens in the cloud on a big instance.
 */
export class ContainerBuild extends Construct {
  public readonly project: codebuild.Project;

  constructor(scope: Construct, id: string, props: ContainerBuildProps) {
    super(scope, id);

    const region = cdk.Stack.of(this).region;
    const account = cdk.Stack.of(this).account;
    const autoTrigger = props.autoTrigger ?? true;
    const computeType = props.computeType ?? codebuild.ComputeType.LARGE;
    const arch = props.architecture ?? 'x86_64';

    // Pick the CodeBuild fleet that matches the target image architecture.
    // arm64 → Graviton fleet (builds aarch64/Jetson images natively, no QEMU);
    // x86_64 → standard Ubuntu fleet. Both ship Docker for `docker build`.
    const buildImage = arch === 'arm64'
      ? codebuild.LinuxArmBuildImage.AMAZON_LINUX_2_STANDARD_3_0
      : codebuild.LinuxBuildImage.STANDARD_7_0;

    // AWS CodeBuild does not support local caching (incl. Docker-layer cache) on
    // the X2_LARGE compute tier. The big NGC images run on X2_LARGE, so only
    // attach the layer cache on the smaller tiers that allow it.
    const supportsLocalCache = computeType !== codebuild.ComputeType.X2_LARGE;

    const environmentVariables: Record<string, codebuild.BuildEnvironmentVariable> = {
      AWS_DEFAULT_REGION: { value: region },
      AWS_ACCOUNT_ID: { value: account },
      ECR_REPO_URI: { value: props.repository.repositoryUri },
      IMAGE_ARCH: { value: arch }, // 'x86_64' | 'arm64' — buildspecs use this to pick the platform/target
      ...(props.requiresNgcLogin
        ? { NGC_SECRET_NAME: { value: `${props.projectName}/ngc-api-key` } }
        : {}),
      ...props.environmentVariables,
    };

    this.project = new codebuild.Project(this, 'Project', {
      projectName: `${props.projectName}-${props.imageName}-build`,
      description: `Build the ${props.imageName} container and push it to ECR`,
      source: codebuild.Source.s3({
        bucket: props.sourceAsset.bucket,
        path: props.sourceAsset.s3ObjectKey,
      }),
      environment: {
        buildImage, // x86_64 Ubuntu, or Graviton ARM fleet for arm64 targets
        computeType,
        privileged: true, // required for `docker build`
        environmentVariables,
      },
      buildSpec: codebuild.BuildSpec.fromSourceFilename(props.buildSpecPath),
      timeout: props.timeout ?? cdk.Duration.hours(1),
      // Docker layer cache speeds up re-builds of unchanged layers (where supported).
      cache: supportsLocalCache
        ? codebuild.Cache.local(codebuild.LocalCacheMode.DOCKER_LAYER)
        : undefined,
    });

    // --- Permissions ---
    props.repository.grantPullPush(this.project);
    props.sourceAsset.grantRead(this.project);

    // ecr:GetAuthorizationToken is account-wide (not resource-scoped) — grantPullPush
    // covers the repo actions but not the auth token, so add it explicitly.
    this.project.addToRolePolicy(new iam.PolicyStatement({
      actions: ['ecr:GetAuthorizationToken'],
      resources: ['*'],
    }));

    if (props.requiresNgcLogin) {
      this.project.addToRolePolicy(new iam.PolicyStatement({
        actions: ['secretsmanager:GetSecretValue'],
        resources: [
          `arn:aws:secretsmanager:${region}:${account}:secret:${props.projectName}/ngc-api-key*`,
        ],
      }));
    }

    if (props.requiresDlcLogin) {
      // AWS Deep Learning Containers live in a per-region AWS-owned account
      // (763104351884 in the standard partition). grantPullPush only covers our
      // own repo, so add image-read on the DLC repos explicitly — otherwise the
      // base-image pull 403s. (GetAuthorizationToken is already account-wide above.)
      this.project.addToRolePolicy(new iam.PolicyStatement({
        actions: ['ecr:BatchGetImage', 'ecr:GetDownloadUrlForLayer', 'ecr:BatchCheckLayerAvailability'],
        resources: [`arn:aws:ecr:${region}:763104351884:repository/*`],
      }));
    }

    // --- Auto-trigger on deploy (re-runs when source OR build inputs change) ---
    if (autoTrigger) {
      // The physical resource ID embeds the asset's S3 object key (a content
      // hash of the repo source) AND a hash of the build's environment
      // variables. CloudFormation only invokes the custom resource (and thus
      // startBuild) when that ID changes. Including the env-var hash means an
      // env-only change — e.g. bumping COSMOS_REF to build a new Cosmos version
      // — also re-triggers the build, even though no source file changed.
      const envFingerprint = JSON.stringify(
        Object.entries(environmentVariables)
          .map(([k, v]) => [k, (v as codebuild.BuildEnvironmentVariable).value])
          .sort(),
      );
      const envHash = createHash('sha256').update(envFingerprint).digest('hex').slice(0, 12);
      const physicalId = cr.PhysicalResourceId.of(
        `${this.project.projectName}-${props.sourceAsset.s3ObjectKey}-${envHash}`,
      );
      const trigger = new cr.AwsCustomResource(this, 'AutoTrigger', {
        resourceType: 'Custom::ContainerBuildTrigger',
        policy: cr.AwsCustomResourcePolicy.fromStatements([
          new iam.PolicyStatement({
            actions: ['codebuild:StartBuild'],
            resources: [this.project.projectArn],
          }),
        ]),
        // codebuild.startBuild is in Lambda's built-in SDK — no need to npm-install
        // a newer one on every trigger (faster, and works in locked-down networks).
        installLatestAwsSdk: false,
        timeout: cdk.Duration.minutes(5), // fire-and-forget: we don't wait for the build
        onCreate: {
          service: 'CodeBuild',
          action: 'startBuild',
          parameters: { projectName: this.project.projectName },
          physicalResourceId: physicalId,
        },
        onUpdate: {
          service: 'CodeBuild',
          action: 'startBuild',
          parameters: { projectName: this.project.projectName },
          physicalResourceId: physicalId,
        },
      });
      trigger.node.addDependency(this.project);
    }
  }

  /**
   * Helper to build the shared source asset once per stack. Excludes build
   * artifacts, deps, and secrets so the upload stays small and reproducible.
   */
  static sourceAsset(scope: Construct, id: string): s3assets.Asset {
    return new s3assets.Asset(scope, id, {
      // constructs/ -> lib/ -> cdk/ -> repo root
      path: path.join(__dirname, '..', '..', '..'),
      exclude: [
        '.git',
        'node_modules',
        'cdk/node_modules',
        'cdk/cdk.out',
        'cdk/cdk.out2',
        'cdk/*.js',
        'cdk/*.d.ts',
        '**/__pycache__',
        '**/*.pyc',
        '.venv',
        'venv',
        '.DS_Store',
        '.env',
        'training/data/**', // large Git LFS datasets — not needed to build images
      ],
    });
  }
}
