import subprocess
import schedule

def run():
    print(f'Run Iteration')
    subprocess.call('python raw_compress_autoencoder_train.py', shell=True)
    subprocess.call('python raw_lal_generate.py', shell=True)
    subprocess.call('python tsms_lal_train.py', shell=True)
    subprocess.call('python tsms_user_vectors_gen.py', shell=True)
    print('End Iteration')

if __name__ == '__main__':

    schedule.every(1).day.at('13:00').do(run)
    
    while True:
        schedule.run_pending()
