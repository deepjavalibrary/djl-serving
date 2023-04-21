#!/bin/bash

< /proc/meminfo grep MemTotal
total=3600
retry=0
while true; do
	< /proc/meminfo grep MemAvailable
	sleep 5
	if [[ "$retry" -ge "$total" ]]; then
   		echo "Monitoring time ended."
    		exit 0
  	fi
	((++retry))
done
