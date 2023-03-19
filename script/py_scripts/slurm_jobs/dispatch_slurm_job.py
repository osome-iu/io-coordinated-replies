import argparse
import os
import json

import helper.strategy_helper as st
import helper.helper as hp
import helper.file_helper as file_hp
import config.config as config
import helper.pandas_helper as pd_hp


def create_and_run_slurm_job(job_name,
                             command,
                             slurm_path
                            ):
    '''
    Creates a slurm script
    :param job_name: Name of job
    :param command: Commnd to be executed
    :param slurm_path: path where slurm script to be saved
    '''
    script = \
f'''#!/bin/bash -i

#SBATCH --job-name={job_name}
#SBATCH -p general
#SBATCH -A general
#SBATCH -o {job_name}_%j.txt
#SBATCH -e {job_name}_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=potem@iu.edu
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=2 # Request that ncpus be allocated per process.
#SBATCH --mem=150GB


#Load any modules that your program needs
source /N/u/potem/Quartz/miniconda3/etc/profile.d/conda.sh
conda activate

#cd /N/u/potem/Quartz/sbatch/multiple_jobs/logs
#Run your program
srun {command}
        '''
    new_slurm_path = os.path.join(slurm_path, f'{job_name}.sh')

    with open(new_slurm_path, 'w') as f:
        f.write(script)
        
    return new_slurm_path


def despatch_single_job(slurm_script, logs_path=None):
    '''
    Despatch the slurm script
    :param slurm_script: slurm script file with path
    :param logs_path: path where slurm script is to be executed
    '''
    command = f'sbatch {slurm_script}'

    if logs_path is not None:
        os.chdir(logs_path)

    os.system(command)
    
    
def parse_args():
    '''
    Parses the arguments
    
    :return arguments passed in command
    '''
    parser = argparse.ArgumentParser(description='Parse zipped campaign data')
    
    parser.add_argument('--slurm_path',
                        dest='slurm_path',
                        help='Path to slurm script')
    
    parser.add_argument('--python_script',
                        dest='python_script',
                        help='Path to python script')
    
    parser.add_argument('--job_name',
                        dest='job_name',
                        help='Name of job')
    
    parser.add_argument('--logs_path',
                        dest='logs_path',
                        help='Place to save logs')
    
    return parser.parse_args()

def main(args):
    python_script = args.python_script
    job_name = args.job_name
    command = f'python {python_script} '
    slurm_path = args.slurm_path
    
    slurm_script = create_and_run_slurm_job(job_name, 
                                            command, 
                                            slurm_path)

    if args.logs_path == None:
        file_hp.create_folder(slurm_path, 'logs')
        logs_path = os.path.join(slurm_path, 'logs')
    else:
        logs_path = args.logs_path

    despatch_single_job(slurm_script, logs_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    
    
    
# python /geode2/home/u070/potem/Quartz/project/infoOps-strategy/script/py_scripts/slurm_jobs/dispatch_slurm_job.py --slurm_path=/N/u/potem/Quartz/sbatch --python_script="/geode2/home/u070/potem/Quartz/project/infoOps-strategy/script/py_scripts/get_reply_embedding.py --type_of=poster" --job_name=poster_twt_emb