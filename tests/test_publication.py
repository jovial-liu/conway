import importlib.util
from pathlib import Path
import subprocess
import json
import pytest

spec=importlib.util.spec_from_file_location('publisher',Path(__file__).resolve().parents[1]/'scripts/publish_hf.py')
publisher=importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def test_only_tracked_small_source_files_are_staged(tmp_path):
    root=tmp_path/'repo'
    root.mkdir()
    subprocess.run(['git','init',str(root)],check=True,capture_output=True)
    for name,text in {'README.md':'code','src/conway/a.py':'x=1','experiments/huge.csv':'private experiment',
                      '.env':'private local config','notes.txt':'personal','docs/INSTALL.md':'install'}.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    subprocess.run(['git','add','.'],cwd=root,check=True)
    subprocess.run(['git','-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','fixture'],cwd=root,check=True,capture_output=True)
    (root/'src/conway/untracked.py').write_text('not included')
    target=tmp_path/'stage';target.mkdir()
    manifest=publisher.stage_snapshot(root,target)
    assert set(manifest['files'])=={'README.md','src/conway/a.py','docs/INSTALL.md'}
    assert not (target/'experiments').exists() and not (target/'.env').exists()
    assert not (target/'src/conway/untracked.py').exists()
    assert json.loads((target/'_source_commit.json').read_text())==manifest
