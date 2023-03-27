import argparse
import os
import json

import helper.strategy_helper as st
import helper.helper as hp
import helper.file_helper as file_hp


def create_slurm_script(job_name,
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
    

def creat_and_despatch_job(python_script,
                           job_name,
                           slurm_path,
                           despatch=False,
                           logs_path=None
                          ):
    '''
    Creates a slurm script and dispatch a job
    :param python_script: python command to run
    :param job_name: name to job
    :param slurm_path: path where slurm script is to be saved
    :param logs_path: path where logs are to be saved
    '''
    command = f'python {python_script} '
    
    slurm_script = create_slurm_script(job_name, 
                                       command, 
                                       slurm_path)

    if despatch == False:
        return slurm_script
    
    if logs_path == None:
        file_hp.create_folder(slurm_path, 'logs')
        logs_path = os.path.join(slurm_path, 'logs')

    despatch_single_job(slurm_script, logs_path)
    
    
    
def cancel_jobs(start_no, end_no):
    '''
    Cancels the jobs
    :param start_no: starting job no
    :param end_no: ending job no
    '''
    for i in range(start_no, end_no+1):
        # id = start_no + i

        command = f'scancel -u potem {i}'

        os.system(command)
        
        
def check_if_error_terminated(error_path):
    '''
    Checks if the job is terminated due to error
    :param error_path: path where error file is present
    
    :return Boolean
    '''
    error_string = file_hp.read_file(error_path)[-1]
    
    return 'Terminated' in error_string