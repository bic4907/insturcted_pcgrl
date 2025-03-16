from invoke import task


@task
def test(c):
    c.run(
        "memlimit 110G rye run python3 -m data.extract_scripts.extract buffer_path=./pcgrl_buffer"
        # "rye run python3 train.py n_envs=600 instruct_freq=1 representation=turtle seed=1 model=nlpconv problem=dungeon3 overwrite=True embed_type=albert aug_type=test instruct=scn-1_se-66"
        # "rye run python3 train.py n_envs=600 instruct_freq=1 representation=turtle seed=1 model=conv problem=dungeon3 overwrite=True embed_type=albert aug_type=test"
    )


@task
def extract_from_runs(c):
    c.run(
        "rye run python3 extract_runs_from_project.py --project_name=sa_err,train_encoder"
    )


@task
def kill(c):
    c.run("kill $(pgrep python3)")
