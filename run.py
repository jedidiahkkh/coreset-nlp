from script import train

import submitit

log_folder = "output_logs/%j"
executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(
    timeout_min=20,
    job_name="part-iii",
    slurm_partition="ampere",
    account="COMPUTERLAB-SL3-GPU",
    # nodes=1,
    # gpus_per_node=1,
    gres="gpu:1",
    mail_type="ALL",
)

print(executor.cluster)

job = executor.submit(train, reduced=executor.cluster != "local")
print(job.job_id)

output = job.result()
print(output)
