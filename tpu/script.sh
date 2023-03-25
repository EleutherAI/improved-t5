gcloud compute tpus tpu-vm scp ./setup.sh t5x2: --worker=all --zone us-central2-b
gcloud compute tpus tpu-vm scp ./run.sh t5x2: --worker=all --zone us-central2-b
gcloud compute tpus tpu-vm ssh test --worker=all --zone us-central2-b --command " bash setup.sh "
gcloud compute tpus tpu-vm ssh test --worker=all --zone us-central2-b --command " bash run.sh > /tmp/out.log"

# gcloud compute tpus tpu-vm scp --recurse ./configs t5x2:~/code --worker=all --zone us-central2-b

# gcloud compute tpus tpu-vm ssh t5x2 --worker=all --zone us-central2-b --command "$ sudo rm /tmp/libtpu_lockfile"

