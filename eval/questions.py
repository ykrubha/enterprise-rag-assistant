"""Hand-written evaluation questions over the Kubernetes concept docs.

Questions are phrased the way a person would ask them, avoiding the docs' wording where
practical. ``fact`` is a regex for the key fact a correct answer must contain, and a test
checks that it really appears in the gold document.

``DEV`` is for development and tuning. It includes an earlier test set, whose failures were
studied, so it can no longer count as held out. ``TEST`` lives in test_questions.py. It was
written after the system was frozen and is only used for reported scores.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Q:
    split: str
    text: str
    gold: tuple[str, ...] = ()
    fact: str | None = None

    @property
    def answerable(self) -> bool:
        return bool(self.gold)


def _a(split: str, text: str, gold: str | tuple[str, ...], fact: str) -> Q:
    return Q(split, text, (gold,) if isinstance(gold, str) else gold, fact)


DEV = [
    _a("dev", "Where does the cluster keep all of its state?", "overview/components.md", r"etcd"),
    _a("dev", "Which component picks the machine a new pod lands on?",
       ("overview/components.md", "scheduling-eviction/kube-scheduler.md"), r"kube-scheduler"),
    _a("dev", "How do machines tell the control plane they are still alive?",
       ("architecture/leases.md", "architecture/nodes.md"), r"lease"),
    _a("dev", "How do I give workloads time to stop cleanly when a machine powers off?",
       "cluster-administration/node-shutdown.md", r"shutdownGracePeriod"),
    _a("dev", "How does deleting a parent object also clean up the things it created?",
       "architecture/garbage-collection.md", r"owner ?references?"),
    _a("dev", "If I don't set an image pull setting, what is used by default?",
       "containers/images.md", r"IfNotPresent"),
    _a("dev", "Can I run a command right before a container is terminated?",
       "containers/container-lifecycle-hooks.md", r"PreStop"),
    _a("dev", "How do I run some workloads under a sandboxed runtime such as gVisor or Kata?",
       "containers/runtime-class.md", r"RuntimeClass"),
    _a("dev", "How do I automatically add replicas when CPU usage climbs?",
       ("workloads/autoscaling/horizontal-pod-autoscale.md", "workloads/autoscaling.md"),
       r"HorizontalPodAutoscaler"),
    _a("dev", "How do I run a helper container for the pod's whole life alongside the main app?",
       "workloads/pods/sidecar-containers.md", r"restartPolicy"),
    _a("dev", "How can separate teams share one cluster with their own isolated names?",
       "overview/working-with-objects/namespaces.md", r"namespaces?"),
    _a("dev", "How do I pick out a group of objects using tags I put on them?",
       "overview/working-with-objects/labels.md", r"selectors?"),
    _a("dev", "Where does kubectl find which clusters and users to connect with?",
       "configuration/organize-cluster-access-kubeconfig.md", r"kubeconfig"),
    _a("dev", "What identity does a workload use when it calls the API server?",
       "security/service-accounts.md", r"service ?account"),
    _a("dev", "How can I add my own object types to the API?",
       "extend-kubernetes/api-extension/custom-resources.md", r"CustomResourceDefinition"),
    Q("dev", "What is the office wifi password?"),
    Q("dev", "How many vacation days do new hires get?"),
    Q("dev", "How do I set up dependency caching in GitHub Actions?"),
    Q("dev", "What caused last month's production incident in checkout?"),
    Q("dev", "What does Google Cloud charge for network egress to Asia?"),
    Q("dev", "How do I rotate our Okta API tokens?"),
]

EARLIER = [
    # Workloads
    _a("dev", "My app needs a one-off setup step to finish before it starts. What should I use?",
       "workloads/pods/init-containers.md", r"init containers?"),
    _a("dev", "How do I stop a scheduled task from starting while its previous run is still going?",
       "workloads/controllers/cron-jobs.md", r"Forbid"),
    _a("dev", "How can finished batch tasks be removed automatically after a while?",
       ("workloads/controllers/ttlafterfinished.md", "workloads/controllers/job.md"),
       r"ttlSecondsAfterFinished"),
    _a("dev", "What should I use to run one log collector on every machine?",
       "workloads/controllers/daemonset.md", r"DaemonSet"),
    _a("dev", "A database needs stable names and its own disk for each replica. Which workload type fits?",
       "workloads/controllers/statefulset.md", r"StatefulSet"),
    _a("dev", "What are the default surge and unavailable percentages during a rolling update?",
       "workloads/controllers/deployment.md", r"25%"),
    _a("dev", "A release went badly. How do I go back to the previous version?",
       "workloads/controllers/deployment.md", r"rollout undo"),
    _a("dev", "How do I make sure node maintenance doesn't take down too many replicas at once?",
       "workloads/pods/disruptions.md", r"PodDisruptionBudget"),
    _a("dev", "What quality of service class does a pod get if requests equal limits for every container?",
       "workloads/pods/pod-qos.md", r"Guaranteed"),
    _a("dev", "What status do I see when a container keeps failing and restart delays keep growing?",
       "workloads/pods/pod-lifecycle.md", r"CrashLoopBackOff"),
    _a("dev", "How do I troubleshoot a running pod whose image has no shell?",
       "workloads/pods/ephemeral-containers.md", r"debug"),
    _a("dev", "Which health check decides whether a container should receive traffic?",
       ("workloads/pods/probes.md", "workloads/pods/pod-lifecycle.md"), r"readiness"),
    # Networking
    _a("dev", "How do I reach my app on a fixed port of each machine without a cloud load balancer?",
       "services-networking/service.md", r"NodePort"),
    _a("dev", "What range of ports is used by default for ports opened on every machine?",
       "services-networking/service.md", r"30000-32767"),
    _a("dev", "How can clients get the backend pod IPs directly, with no virtual IP in front?",
       "services-networking/service.md", r"headless|clusterIP.{0,20}None"),
    _a("dev", "With no rules in place, are pods restricted in who they can talk to?",
       "services-networking/network-policies.md", r"non-isolated"),
    _a("dev", "How do I send web traffic for different domain names to different backends?",
       "services-networking/ingress.md", r"name.based"),
    _a("dev", "What is the newer alternative to Ingress for routing traffic?",
       ("services-networking/gateway.md", "services-networking/ingress.md"), r"Gateway"),
    _a("dev", "What DNS name do I use to reach a service in another namespace?",
       "services-networking/dns-pod-service.md", r"svc\.cluster|<namespace>|prod"),
    _a("dev", "How do I keep service traffic in the same availability zone to cut cross-zone costs?",
       "services-networking/topology-aware-routing.md", r"topology.mode|Topology Aware"),
    # Storage
    _a("dev", "Which volume mode lets just one machine mount a disk for reading and writing?",
       "storage/persistent-volumes.md", r"ReadWriteOnce"),
    _a("dev", "How do I keep a disk's data after its claim gets deleted?",
       "storage/persistent-volumes.md", r"Retain"),
    _a("dev", "What gives containers in a pod scratch space that vanishes with the pod?",
       ("storage/volumes.md", "storage/ephemeral-volumes.md"), r"emptyDir"),
    _a("dev", "How can disks be created on demand with no admin creating them first?",
       "storage/dynamic-provisioning.md", r"StorageClass"),
    _a("dev", "How do I take a point-in-time copy of a disk?",
       "storage/volume-snapshots.md", r"VolumeSnapshot"),
    # Configuration and security
    _a("dev", "What is the maximum size of data I can put in a ConfigMap?",
       "configuration/configmap.md", r"1 ?MiB"),
    _a("dev", "Are credentials stored as Secrets encrypted by default?",
       "configuration/secret.md", r"unencrypted"),
    _a("dev", "What happens when a container goes over its memory cap, compared with its CPU cap?",
       "configuration/manage-resources-containers.md", r"throttl|OOM|out of memory"),
    _a("dev", "Which pod security level is the most locked down?",
       "security/pod-security-standards.md", r"Restricted"),
    _a("dev", "What modes can namespace pod security checks run in, such as blocking or just warning?",
       "security/pod-security-admission.md", r"enforce"),
    _a("dev", "What is the general principle to follow when giving out cluster permissions?",
       "security/rbac-good-practices.md", r"least privilege"),
    _a("dev", "How do I cap the total CPU and memory one team's namespace can use?",
       "policy/resource-quotas.md", r"ResourceQuota"),
    _a("dev", "How do I give containers default CPU and memory requests when they don't set any?",
       "policy/limit-range.md", r"LimitRange"),
    # Scheduling
    _a("dev", "How do I keep ordinary workloads off machines reserved for GPU jobs?",
       "scheduling-eviction/taint-and-toleration.md", r"taints?"),
    _a("dev", "Which setting removes already-running pods from a machine that gets marked as off limits?",
       "scheduling-eviction/taint-and-toleration.md", r"NoExecute"),
    _a("dev", "How do I distribute replicas evenly across zones?",
       "scheduling-eviction/topology-spread-constraints.md", r"maxSkew|topologySpreadConstraints"),
    _a("dev", "How do I make a pod run only on machines with SSDs?",
       "scheduling-eviction/assign-pod-node.md", r"nodeSelector|node affinity"),
    _a("dev", "Can an important workload push out less important ones when the cluster is full?",
       "scheduling-eviction/pod-priority-preemption.md", r"preempt"),
    _a("dev", "What signal triggers pod removal when a machine is running out of RAM?",
       "scheduling-eviction/node-pressure-eviction.md", r"memory\.available"),
    _a("dev", "How does the kubelet decide which containers can use swap?",
       "cluster-administration/swap-memory-management.md", r"LimitedSwap"),
    # Not in the knowledge base
    Q("dev", "What is our company's parental leave policy?"),
    Q("dev", "How much would a three-node EKS cluster cost per month?"),
    Q("dev", "What was our revenue last quarter?"),
    Q("dev", "Who is the on-call engineer this week?"),
    Q("dev", "How do I configure Terraform remote state in S3?"),
    Q("dev", "What is the best autovacuum setting for a 2 TB Postgres table?"),
    Q("dev", "What is the syntax for a Python list comprehension?"),
    Q("dev", "How do I reset my VPN password?"),
    Q("dev", "Which Helm chart version does the payments team deploy?"),
    Q("dev", "What eviction policy does our Redis cache use?"),
    Q("dev", "How do I submit an expense report?"),
    Q("dev", "What is the maximum timeout for an AWS Lambda function?"),
]


DEV = DEV + EARLIER
