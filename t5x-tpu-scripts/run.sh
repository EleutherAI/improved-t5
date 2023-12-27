#pkill train.py
#rm -f /tmp/libtpu_lockfile

# --worker=all
gcloud compute tpus tpu-vm ssh $1 \
    --worker=all \
    --zone us-central2-b \
    --command "$2"
