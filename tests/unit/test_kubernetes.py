import copy
import subprocess

import pytest

from app.analyzer.project_scanner import ProjectScanner
from app.kubernetes.generator import (K8sProtectionError, apply_instructions, generate_manifests, k8s_name,
                                      validate_manifests, write_manifests)
from tests.unit.test_compose import APP

yaml = pytest.importorskip("yaml")

DF = "FROM python:3.13-slim\nRUN useradd -u 10001 app\nUSER 10001\nEXPOSE 5000\nCMD [\"python\",\"app.py\"]\n"


def build(make_project, settings, df=DF, **kw):
    root = make_project({"requirements.txt": "flask\npsycopg2-binary\nredis\n", "app.py": APP, "Dockerfile": df})
    a = ProjectScanner(settings).scan(root)
    return root, a, generate_manifests(a, dockerfile=df, **kw)


def by_kind(res):
    return {d["kind"]: d for d in res.docs}


def test_full_set_and_round_trip(make_project, settings):
    _, _, res = build(make_project, settings, image="registry.example.com/shop:1.0.0", ingress_host="shop.example.com")
    k = by_kind(res)
    assert set(k) == {"Deployment", "Service", "Secret", "Ingress", "HorizontalPodAutoscaler"}
    for rel, text in res.files.items():  # every emitted file parses back to the same document
        assert yaml.safe_load(text) == res.sets[0].file_map()[rel]
    assert list(yaml.safe_load_all(res.combined)) == res.docs
    v = validate_manifests(res.docs, use_kubectl=False)
    assert v.ok, v.errors


def test_deployment_hardening_and_probes(make_project, settings):
    _, _, res = build(make_project, settings)
    dep = by_kind(res)["Deployment"]
    c = dep["spec"]["template"]["spec"]["containers"][0]
    pod = dep["spec"]["template"]["spec"]
    assert c["securityContext"] == {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}}
    assert pod["securityContext"]["runAsNonRoot"] is True and pod["securityContext"]["runAsUser"] == 10001
    assert pod["automountServiceAccountToken"] is False
    assert c["resources"]["requests"] and c["resources"]["limits"]
    assert all(p in c for p in ("startupProbe", "readinessProbe", "livenessProbe")) and "tcpSocket" in c["livenessProbe"]
    assert c["ports"][0]["containerPort"] == 5000 and dep["spec"]["replicas"] == 2


def test_http_probe_when_health_path_given(make_project, settings):
    _, _, res = build(make_project, settings, health_path="/healthz")
    c = by_kind(res)["Deployment"]["spec"]["template"]["spec"]["containers"][0]
    assert c["readinessProbe"]["httpGet"] == {"path": "/healthz", "port": "http"}


def test_secret_template_has_placeholders_only(make_project, settings):
    _, _, res = build(make_project, settings)
    sec = by_kind(res)["Secret"]
    assert set(sec["stringData"]) >= {"SECRET_KEY", "DATABASE_URL", "REDIS_URL"}
    assert set(sec["stringData"].values()) == {"REPLACE_ME"}
    bad = copy.deepcopy(res.docs)
    next(d for d in bad if d["kind"] == "Secret")["stringData"]["SECRET_KEY"] = "s3cr3t-value"
    assert not validate_manifests(bad, use_kubectl=False).ok


def test_backing_services_not_generated(make_project, settings):
    _, _, res = build(make_project, settings)
    assert not any("postgres" in d["metadata"]["name"] for d in res.docs)
    assert any("NOT generated" in n for n in res.notes)


def test_named_user_warns_and_no_enforcement(make_project, settings):
    _, _, res = build(make_project, settings, df="FROM python:3.13-slim\nUSER app\nEXPOSE 5000\n")
    pod = by_kind(res)["Deployment"]["spec"]["template"]["spec"]
    assert "runAsNonRoot" not in pod["securityContext"] and any("numeric" in w for w in res.warnings)


def test_volume_creates_pvc_single_replica_no_hpa(make_project, settings):
    _, _, res = build(make_project, settings, df=DF + 'VOLUME ["/data"]\n')
    k = by_kind(res)
    assert "PersistentVolumeClaim" in k and "HorizontalPodAutoscaler" not in k
    assert k["Deployment"]["spec"]["replicas"] == 1 and k["Deployment"]["spec"]["strategy"] == {"type": "Recreate"}
    assert validate_manifests(res.docs, use_kubectl=False).ok


def test_namespace_and_unpinned_image_warning(make_project, settings):
    _, _, res = build(make_project, settings, namespace="shop", image="shop:latest")
    assert all(d["metadata"]["namespace"] == "shop" for d in res.docs)
    assert any("unpinned" in w for w in res.warnings)


def test_validator_catches_structural_problems(make_project, settings):
    _, _, res = build(make_project, settings, ingress_host="h.example.com")
    docs = copy.deepcopy(res.docs)
    k = {d["kind"]: d for d in docs}
    k["Deployment"]["spec"]["template"]["metadata"]["labels"] = {"x": "y"}
    k["Service"]["spec"]["selector"] = {"nope": "1"}
    k["Ingress"]["spec"]["rules"][0]["http"]["paths"][0]["backend"]["service"]["name"] = "ghost"
    k["HorizontalPodAutoscaler"]["spec"]["scaleTargetRef"]["name"] = "ghost"
    c = k["Deployment"]["spec"]["template"]["spec"]["containers"][0]
    c["securityContext"]["privileged"] = True
    c["resources"]["requests"]["memory"] = "2Gi"
    k["Deployment"]["spec"]["template"]["spec"]["hostNetwork"] = True
    k["Deployment"]["spec"]["template"]["spec"]["volumes"] = [{"name": "h", "hostPath": {"path": "/"}}]
    text = " ".join(validate_manifests(docs, use_kubectl=False).errors)
    for frag in ("selector does not match", "matches no Deployment", "does not exist", "privileged", "requests exceed",
                 "hostNetwork", "hostPath"):
        assert frag in text, frag


def test_invalid_names_and_credentials():
    docs = [{"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "Bad_Name"},
             "data": {"URL": "postgresql://app:hunter2@db/x"}}]
    t = " ".join(validate_manifests(docs, use_kubectl=False).errors)
    assert "DNS-1123" in t and "credential" in t


def test_kubectl_dry_run_is_client_side_and_optional(make_project, settings):
    _, _, res = build(make_project, settings)
    calls = []

    def runner(cmd, **k):
        calls.append((cmd, k))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    v = validate_manifests(res.docs, runner=runner, which=lambda n: "/usr/bin/kubectl", yaml_text=res.combined)
    assert v.kubectl_checked and "--dry-run=client" in calls[0][0] and "apply" in calls[0][0] and "create" not in calls[0][0]
    absent = validate_manifests(res.docs, which=lambda n: None, yaml_text=res.combined)
    assert absent.ok and not absent.kubectl_checked and any("not installed" in w for w in absent.warnings)
    fail = lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", "error: invalid")  # noqa: E731
    assert not validate_manifests(res.docs, runner=fail, which=lambda n: "x", yaml_text=res.combined).ok


def test_instructions_are_text_only(make_project, settings):
    _, _, res = build(make_project, settings, namespace="shop")
    text = "\n".join(apply_instructions(res, namespace="shop"))
    assert "kubectl apply -f k8s -R -n shop" in text and "create namespace shop" in text and "REPLACE_ME" in text


def test_monorepo_one_set_per_component(make_project, settings):
    files = {"backend/requirements.txt": "flask\n", "backend/app.py": APP, "worker/requirements.txt": "flask\n",
             "worker/app.py": APP}
    root = make_project(files)
    a = ProjectScanner(settings).scan(root)
    res = generate_manifests(a, dockerfile=DF)
    assert {s.component for s in res.sets} == {"backend", "worker"}
    names = [d["metadata"]["name"] for d in res.docs if d["kind"] == "Deployment"]
    assert len(set(names)) == 2 and validate_manifests(res.docs, use_kubectl=False).ok


def test_no_port_raises(make_project, settings):
    root = make_project({"README.md": "x"})
    with pytest.raises(ValueError):
        generate_manifests(ProjectScanner(settings).scan(root))


def test_write_protection_and_symlink(make_project, settings, tmp_path):
    root, _, res = build(make_project, settings)
    paths = write_manifests(root, res)
    assert all(p.is_relative_to(root / "k8s") for p in paths) and (root / "k8s" / "proj" / "deployment.yaml").exists()
    with pytest.raises(K8sProtectionError):
        write_manifests(root, res)
    write_manifests(root, res, overwrite=True)
    assert (root / "k8s" / "proj" / "deployment.yaml.bak").exists()
    with pytest.raises(PermissionError):
        write_manifests(root, res, directory="../x", overwrite=True)
    (tmp_path / "victim").write_text("keep")
    (root / "k8s" / "proj" / "service.yaml").unlink()
    (root / "k8s" / "proj" / "service.yaml").symlink_to(tmp_path / "victim")
    with pytest.raises(PermissionError):
        write_manifests(root, res, overwrite=True)
    assert (tmp_path / "victim").read_text() == "keep"


def test_k8s_name():
    assert k8s_name("My_App v2!") == "my-app-v2" and k8s_name("___") == "app"


def test_cli_k8s_prints_writes_and_never_deploys(make_project, settings, capsys, monkeypatch):
    from app.main import main
    root, _, _ = build(make_project, settings)
    monkeypatch.setenv("HISTORY_DB", str(root.parent / "h.db"))
    assert main(["k8s", str(root), "--image", "registry.example.com/shop:1.0.0"]) == 0
    out = capsys.readouterr()
    assert "kind: Deployment" in out.out and "Nothing was deployed" in out.err and "kubectl apply -f k8s" in out.err
    assert not (root / "k8s").exists()
    assert main(["k8s", str(root), "--write"]) == 0 and (root / "k8s" / "proj" / "deployment.yaml").exists()
    assert main(["k8s", str(root), "--write"]) == 3
    assert main(["k8s", str(root), "--write", "--yes-overwrite"]) == 0


def test_k8s_uses_dockerfile_expose_and_works_without_detected_port(make_project, settings):
    root = make_project({"requirements.txt": "flask\n", "app.py": "from flask import Flask\napp = Flask(__name__)\n"})
    a = ProjectScanner(settings).scan(root)
    a.primary_unit.ports = []
    for u in a.units:
        u.ports = []
    res = generate_manifests(a, dockerfile="FROM python:3.13-slim\nUSER 10001\nEXPOSE 8081\n")
    assert by_kind(res)["Service"]["spec"]["ports"][0]["port"] == 8081
    res2 = generate_manifests(ProjectScanner(settings).scan(root), dockerfile="FROM python:3.13-slim\nUSER 10001\nEXPOSE 9000\n")
    assert by_kind(res2)["Service"]["spec"]["ports"][0]["port"] == 9000 and any("EXPOSEs 9000" in w for w in res2.warnings)
