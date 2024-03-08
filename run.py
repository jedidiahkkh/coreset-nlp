import logging

from script import train

import submitit

logging.basicConfig(
    format="%(levelname)s:%(name)s:%(asctime)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger()

log_folder = "output_logs/%j"
executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(
    timeout_min=12 * 60,
    job_name="part-iii",
    slurm_partition="ampere",
    account="COMPUTERLAB-SL3-GPU",
    # nodes=1,
    # gpus_per_node=1,
    gres="gpu:1",
    mail_type="ALL",
)

logger.info(f"Executing on {executor.cluster}")

job = executor.submit(train, reduced=executor.cluster != "slurm")
logger.info("Job submitted with job id %s", job.job_id)

output = job.result()
logger.info(f"Output: {output}")
