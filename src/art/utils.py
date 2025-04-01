from .types import Message
import time
import asyncio
import functools
import logging
import inspect
import subprocess
from typing import Callable, TypeVar, Optional, Any, Union, cast, Coroutine


def format_message(message: Message) -> str:
    """Format a message into a readable string."""
    # Format the role and content
    role = message["role"].capitalize()
    content = message.get("content", message.get("refusal", "")) or ""

    # Format any tool calls
    tool_calls_text = "\n" if content else ""
    tool_calls_text += "\n".join(
        f"{tool_call['function']['name']}({tool_call['function']['arguments']})"
        for tool_call in message.get("tool_calls") or []
    )

    # Combine all parts
    formatted_message = f"{role}:\n{content}{tool_calls_text}"
    return formatted_message


T = TypeVar("T")
R = TypeVar("R")


def retry(
    max_attempts: int = 3,
    delay: float = 0.25,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[
    [Callable[..., Union[T, Coroutine[Any, Any, R]]]],
    Callable[..., Union[T, Coroutine[Any, Any, R]]],
]:
    """
    Retry decorator with exponential backoff for both synchronous and asynchronous functions.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay between retries
        exceptions: Tuple of exceptions to catch and retry on
        on_retry: Optional callback function called on retry with the exception and attempt number

    Returns:
        Decorated function that will retry on specified exceptions
    """

    def decorator(func):
        is_coroutine = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        raise

                    if on_retry:
                        on_retry(e, attempt)
                    else:
                        logging.warning(
                            f"Retry {attempt}/{max_attempts} for {func.__name__} "
                            f"after error: {str(e)}"
                        )

                    time.sleep(current_delay)
                    current_delay *= backoff_factor

            assert last_exception is not None
            raise last_exception

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> R:
            last_exception = None
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        raise

                    if on_retry:
                        on_retry(e, attempt)
                    else:
                        logging.warning(
                            f"Retry {attempt}/{max_attempts} for {func.__name__} "
                            f"after error: {str(e)}"
                        )

                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor

            assert last_exception is not None
            raise last_exception

        return async_wrapper if is_coroutine else sync_wrapper

    return decorator


def free_gpu_memory() -> None:
    """
    Free memory on GPU 0 by finding and killing processes that have claimed memory on it.
    Requires nvidia-smi to be installed and accessible.
    """
    try:
        import re
        import os
        import time
        import sys
        import signal

        # Quick check if GPU memory is already low, exit early if so
        try:
            memory_info = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ],
                text=True,
            ).strip()

            used_memory = int(memory_info)
            logging.info(f"Current GPU 0 memory usage: {used_memory} MiB")

            # If memory usage is already low (less than 100 MiB), exit early
            if used_memory < 100:
                logging.info("GPU memory usage already low, no cleanup needed")
                return
        except Exception as e:
            # If we can't check memory, continue with cleanup process
            logging.warning(f"Could not check initial GPU memory: {str(e)}")

        # Method 1: Use nvidia-smi to get processes that are using GPU 0
        logging.info("Attempting to free GPU 0 memory...")

        # Get detailed GPU info first for diagnostic purposes
        try:
            gpu_info = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=gpu_name,memory.total,memory.used,memory.free",
                    "--format=csv",
                ],
                text=True,
            ).strip()
            logging.info(f"GPU status before cleanup:\n{gpu_info}")
        except Exception as e:
            logging.warning(f"Could not get GPU info: {str(e)}")

        # First, try to get all processes using GPU memory
        nvidia_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        )

        # Extract PIDs from the output that are using GPU 0
        # Format is "pid, gpu_name, used_memory"
        gpu_processes = []
        for line in nvidia_output.strip().split("\n"):
            if line.strip():
                parts = line.split(",")
                if len(parts) >= 3 and (
                    "0" in parts[1] or parts[1].strip() == ""
                ):  # Check if GPU 0 is in the gpu_name or if it's empty (sometimes happens)
                    try:
                        pid = int(parts[0].strip())
                        gpu_processes.append((pid, parts[2].strip()))
                    except ValueError:
                        continue

        if not gpu_processes:
            # Alternative method: Use a more general approach to find processes
            logging.info(
                "No processes found with first method, trying alternative approach..."
            )
            try:
                # Get a full listing with more details
                detailed_output = subprocess.check_output(
                    ["nvidia-smi", "-q", "-i", "0"],
                    text=True,
                )

                # Parse the output to extract process IDs
                process_matches = re.findall(r"Process ID\s+:\s+(\d+)", detailed_output)
                for pid_str in process_matches:
                    try:
                        pid = int(pid_str)
                        gpu_processes.append((pid, "unknown"))
                    except ValueError:
                        continue
            except Exception as e:
                logging.warning(f"Alternative method failed: {str(e)}")

        # Also check for Python processes that might be using GPU
        try:
            # Find all python processes
            python_procs = (
                subprocess.check_output(["pgrep", "python"], text=True)
                .strip()
                .split("\n")
            )
            current_pid = os.getpid()

            for pid_str in python_procs:
                if pid_str.strip() and int(pid_str.strip()) != current_pid:
                    # Don't add our own process
                    pid = int(pid_str.strip())
                    # Check if already in our list
                    if not any(p[0] == pid for p in gpu_processes):
                        gpu_processes.append((pid, "python process"))
        except Exception as e:
            logging.warning(f"Python process check failed: {str(e)}")

        if not gpu_processes:
            logging.info("No processes found using GPU 0")
        else:
            logging.info(
                f"Found {len(gpu_processes)} potential processes using GPU 0. Terminating them..."
            )

            # Kill each process
            for pid, memory_usage in gpu_processes:
                try:
                    # Check if process exists before trying to kill it
                    process_exists = False
                    try:
                        # On Linux
                        process_exists = os.path.exists(f"/proc/{pid}")
                    except:
                        # Try alternative method
                        try:
                            os.kill(
                                pid, 0
                            )  # Signal 0 doesn't kill but checks if process exists
                            process_exists = True
                        except ProcessLookupError:
                            process_exists = False
                        except Exception:
                            # Process exists but might be owned by another user
                            process_exists = True

                    if not process_exists:
                        logging.info(f"Process {pid} no longer exists, skipping")
                        continue

                    # First try a normal termination
                    logging.info(f"Terminating process {pid} using {memory_usage}")
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except:
                        subprocess.run(["kill", str(pid)], check=False)

                    # Give it a moment to terminate
                    time.sleep(0.5)

                    # If still running, force kill
                    try:
                        # Check if still exists
                        os.kill(pid, 0)
                        logging.info(f"Process {pid} still running, force killing")
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except:
                            subprocess.run(["kill", "-9", str(pid)], check=False)
                    except ProcessLookupError:
                        logging.info(f"Process {pid} successfully terminated")
                    except Exception as e:
                        logging.warning(
                            f"Could not check process {pid} status: {str(e)}"
                        )
                        # Try force kill anyway
                        try:
                            subprocess.run(["kill", "-9", str(pid)], check=False)
                        except:
                            pass
                except Exception as e:
                    logging.warning(f"Failed to terminate process {pid}: {str(e)}")

        # Check if we need more aggressive methods by checking current memory usage
        try:
            memory_info = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ],
                text=True,
            ).strip()

            used_memory = int(memory_info)
            logging.info(
                f"GPU 0 memory usage after process termination: {used_memory} MiB"
            )

            # Only try aggressive GPU reset methods if memory is still significantly used
            if used_memory > 100:
                # Try to reset GPU through driver directly - this is more aggressive
                try:
                    logging.info("Attempting to reset GPU directly...")

                    # Method 1: Try nvidia-smi GPU reset
                    try:
                        logging.info("Attempting nvidia-smi --gpu-reset...")
                        subprocess.run(
                            ["nvidia-smi", "--gpu-reset", "-i", "0"], check=False
                        )
                    except Exception as e:
                        logging.warning(f"GPU reset failed: {str(e)}")

                    # Method 2: Try reloading NVIDIA driver module if running as root
                    if os.geteuid() == 0:  # Check if running as root
                        try:
                            logging.info(
                                "Attempting to reload NVIDIA driver modules..."
                            )
                            # Check if rmmod/modprobe commands exist before trying to use them
                            if (
                                subprocess.run(
                                    "which rmmod >/dev/null 2>&1", shell=True
                                ).returncode
                                == 0
                                and subprocess.run(
                                    "which modprobe >/dev/null 2>&1", shell=True
                                ).returncode
                                == 0
                            ):
                                # Unload and reload cuda modules if possible
                                subprocess.run(
                                    "rmmod nvidia_uvm; rmmod nvidia_drm; rmmod nvidia_modeset; rmmod nvidia; "
                                    + "modprobe nvidia; modprobe nvidia_modeset; modprobe nvidia_drm; modprobe nvidia_uvm",
                                    shell=True,
                                    check=False,
                                )
                            else:
                                logging.info(
                                    "rmmod/modprobe commands not found, skipping driver reload"
                                )
                        except Exception as e:
                            logging.warning(f"Driver reload failed: {str(e)}")

                    # Method 3: Try to clear CUDA cache using Python if torch is available
                    try:
                        logging.info(
                            "Attempting to clear CUDA cache if torch is available..."
                        )
                        # Try to import torch without failing if not present
                        if "torch" in sys.modules or any(
                            "torch" in m for m in sys.modules
                        ):
                            import torch

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                logging.info("Cleared PyTorch CUDA cache")
                    except Exception as e:
                        logging.warning(f"PyTorch CUDA cache clearing failed: {str(e)}")

                    # Method 4: If nvidia-persistenced is running, try to stop it
                    try:
                        persistenced_pid = subprocess.check_output(
                            ["pgrep", "nvidia-persiste"], text=True
                        ).strip()
                        if persistenced_pid:
                            logging.info(
                                f"Found nvidia-persistenced (PID: {persistenced_pid}), stopping it..."
                            )
                            subprocess.run(["kill", persistenced_pid], check=False)
                    except Exception:
                        pass  # Ignore if not found

                except Exception as e:
                    logging.warning(f"GPU reset attempts failed: {str(e)}")
            else:
                logging.info(
                    "Memory usage already low, skipping aggressive GPU reset methods"
                )
        except Exception as e:
            logging.warning(
                f"Could not check memory to determine if aggressive methods needed: {str(e)}"
            )

        # Final verification of memory usage
        time.sleep(0.5)  # Shorter wait time for final check
        try:
            memory_info = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ],
                text=True,
            ).strip()
            logging.info(f"Final GPU 0 memory usage: {memory_info} MiB")
        except Exception as e:
            logging.warning(f"Failed to verify memory usage: {str(e)}")

    except FileNotFoundError:
        logging.error(
            "nvidia-smi not found. Make sure NVIDIA drivers are installed and nvidia-smi is in PATH"
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running nvidia-smi: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error while freeing GPU memory: {str(e)}")
