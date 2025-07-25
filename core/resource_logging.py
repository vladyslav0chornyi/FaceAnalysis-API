import threading
import time
import psutil
import platform
from datetime import datetime
import os

def resource_logger(logfile="resource_log.txt", interval=1):
    process = psutil.Process(os.getpid())
    boot_time = datetime.fromtimestamp(psutil.boot_time()).strftime("%Y-%m-%d %H:%M:%S")
    with open(logfile, "w") as f:
        # --- Системна інфа ---
        f.write("# System info\n")
        f.write(f"Datetime: {datetime.now()}\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"Python: {platform.python_version()}\n")
        f.write(f"CPU: {platform.processor()}\n")
        f.write(f"Boot time: {boot_time}\n")
        f.write(f"Logical CPUs: {psutil.cpu_count(logical=True)}\n")
        f.write(f"Physical CPUs: {psutil.cpu_count(logical=False)}\n")
        f.write(f"RAM total (MB): {psutil.virtual_memory().total/1024/1024:.1f}\n")
        f.write(f"Swap total (MB): {psutil.swap_memory().total/1024/1024:.1f}\n")
        f.write("\n# Resource log columns:\n")
        f.write("time_sec\t"
                "cpu%_all\t"
                + "\t".join([f"cpu%_{i}" for i in range(psutil.cpu_count(logical=True))]) + "\t"
                "ram_used_mb\tram_avail_mb\tram_free_mb\tram_total_mb\tram_percent\t"
                "swap_used_mb\tswap_free_mb\tswap_total_mb\tswap_percent\t"
                "proc_ram_mb\tproc_vms_mb\tproc_cpu%\tproc_threads\tproc_handles\t"
                "io_read_mb\tio_write_mb\t"
                "net_sent_mb\tnet_recv_mb\n")
        start = time.time()
        net0 = psutil.net_io_counters()
        io0 = process.io_counters() if hasattr(process, "io_counters") else None

        while True:
            now = time.time()
            cpu = psutil.cpu_percent()
            cpu_each = psutil.cpu_percent(percpu=True)
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            proc = process
            proc_mem = proc.memory_info()
            proc_ram = proc_mem.rss/1024/1024
            proc_vms = proc_mem.vms/1024/1024
            proc_cpu = proc.cpu_percent()
            proc_threads = proc.num_threads()
            proc_handles = proc.num_handles() if hasattr(proc, "num_handles") else 0
            io = proc.io_counters() if hasattr(proc, "io_counters") else None
            read_mb = io.read_bytes/1024/1024 if io else 0
            write_mb = io.write_bytes/1024/1024 if io else 0
            net = psutil.net_io_counters()
            net_sent = (net.bytes_sent-net0.bytes_sent)/1024/1024
            net_recv = (net.bytes_recv-net0.bytes_recv)/1024/1024

            f.write(f"{now-start:.1f}\t"
                    f"{cpu:.1f}\t"
                    + "\t".join([f"{x:.1f}" for x in cpu_each]) + "\t"
                    f"{vm.used/1024/1024:.1f}\t{vm.available/1024/1024:.1f}\t{vm.free/1024/1024:.1f}\t{vm.total/1024/1024:.1f}\t{vm.percent:.1f}\t"
                    f"{swap.used/1024/1024:.1f}\t{swap.free/1024/1024:.1f}\t{swap.total/1024/1024:.1f}\t{swap.percent:.1f}\t"
                    f"{proc_ram:.1f}\t{proc_vms:.1f}\t{proc_cpu:.1f}\t{proc_threads}\t{proc_handles}\t"
                    f"{read_mb:.1f}\t{write_mb:.1f}\t"
                    f"{net_sent:.3f}\t{net_recv:.3f}\n")
            f.flush()
            time.sleep(interval)
            # Зупиняємо якщо main thread завершився
            if not any([t.is_alive() for t in threading.enumerate() if t.name == "MainThread"]):
                break

def start_resource_monitor(logfile="resource_log.txt", interval=1):
    monitor_thread = threading.Thread(target=resource_logger, args=(logfile, interval), daemon=True)
    monitor_thread.start()
    return monitor_thread