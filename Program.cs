using System;
using System.Threading;
using System.Threading.Tasks;

namespace MiningSimulation
{
    class Program
    {
        static int coalDeposit = 2000;
        static int warehouseCoal = 0;
        const int vehicleCapacity = 200;
        const int extractionTime = 10;
        const int unloadTime = 10;
        const int transportTime = 10000;

        static SemaphoreSlim depositSemaphore = new SemaphoreSlim(2, 2);
        static SemaphoreSlim warehouseSemaphore = new SemaphoreSlim(1, 1);
        static object depositLock = new object();
        static object warehouseLock = new object();

        static string[] minerStatus;
        static object consoleLock = new object();
        static bool simulationRunning = true;

        static int numberOfMiners = 1;
        static bool showLiveDisplay = false;

        static void Main(string[] args)
        {
            Console.WriteLine("=== Pomiar przyśpieszenia i efektywności ===\n");

            double baseTime = 0;

            for (numberOfMiners = 1; numberOfMiners <= 6; numberOfMiners++)
            {
                coalDeposit = 2000;
                warehouseCoal = 0;
                simulationRunning = true;
                minerStatus = new string[numberOfMiners];

                for (int i = 0; i < minerStatus.Length; i++)
                {
                    minerStatus[i] = "Oczekuje...";
                }

                DateTime startTime = DateTime.Now;

                Task displayTask = null;
                if (showLiveDisplay && numberOfMiners == 6)
                {
                    Console.Clear();
                    Console.CursorVisible = false;
                    displayTask = Task.Run(() => UpdateDisplay());
                }

                Task[] miners = new Task[numberOfMiners];
                for (int i = 0; i < miners.Length; i++)
                {
                    int minerId = i + 1;
                    miners[i] = Task.Run(() => MinerWork(minerId));
                }

                Task.WaitAll(miners);

                Thread.Sleep(200);

                simulationRunning = false;
                if (displayTask != null)
                {
                    displayTask.Wait();
                }

                DateTime endTime = DateTime.Now;
                double totalTime = (endTime - startTime).TotalSeconds;

                if (numberOfMiners == 1)
                {
                    baseTime = totalTime;
                }

                double speedup = baseTime / totalTime;
                double efficiency = speedup / numberOfMiners;

                if (showLiveDisplay && numberOfMiners == 6)
                {
                    Console.SetCursorPosition(0, 10);
                }
                Console.WriteLine($"liczba górników: {numberOfMiners}, czas: {totalTime:F2} s, " +
                                  $"przyśpieszenie: {speedup:F2}, efektywność: {efficiency:F2}");
            }

            Console.WriteLine("\n=== Pomiary zakończone ===");
            if (showLiveDisplay)
            {
                Console.CursorVisible = true;
            }
        }

        static void UpdateDisplay()
        {
            while (simulationRunning || coalDeposit > 0 || warehouseCoal < 2000)
            {
                lock (consoleLock)
                {
                    Console.SetCursorPosition(0, 0);
                    Console.WriteLine($"Stan złoża: {coalDeposit} jednostek węgla".PadRight(60));
                    Console.WriteLine($"Stan magazynu: {warehouseCoal} jednostek węgla".PadRight(60));
                    Console.WriteLine("".PadRight(60));

                    for (int i = 0; i < minerStatus.Length; i++)
                    {
                        Console.WriteLine($"Górnik {i + 1}: {minerStatus[i]}".PadRight(60));
                    }
                }

                Thread.Sleep(100);
            }
        }

        static void MinerWork(int minerId)
        {
            int minerIndex = minerId - 1;

            while (true)
            {
                lock (depositLock)
                {
                    if (coalDeposit <= 0)
                    {
                        if (showLiveDisplay)
                        {
                            lock (consoleLock)
                            {
                                minerStatus[minerIndex] = "Zakończył pracę.";
                            }
                        }
                        break;
                    }
                }

                if (showLiveDisplay)
                {
                    lock (consoleLock)
                    {
                        minerStatus[minerIndex] = "Wydobywa węgiel...";
                    }
                }
                int extractedCoal = ExtractCoal(minerId);
                
                if (extractedCoal == 0)
                {
                    if (showLiveDisplay)
                    {
                        lock (consoleLock)
                        {
                            minerStatus[minerIndex] = "Zakończył pracę.";
                        }
                    }
                    break;
                }

                if (showLiveDisplay)
                {
                    lock (consoleLock)
                    {
                        minerStatus[minerIndex] = "Transportuje do magazynu...";
                    }
                }
                Thread.Sleep(transportTime);

                if (showLiveDisplay)
                {
                    lock (consoleLock)
                    {
                        minerStatus[minerIndex] = "Rozładowuje węgiel...";
                    }
                }
                UnloadCoal(minerId, extractedCoal);
            }
        }

        static int ExtractCoal(int minerId)
        {
            depositSemaphore.Wait();

            try
            {
                int coalToExtract = 0;

                lock (depositLock)
                {
                    if (coalDeposit <= 0)
                    {
                        return 0;
                    }

                    coalToExtract = Math.Min(vehicleCapacity, coalDeposit);
                    coalDeposit -= coalToExtract;
                }

                Thread.Sleep(coalToExtract * extractionTime);

                return coalToExtract;
            }
            finally
            {
                depositSemaphore.Release();
            }
        }

        static void UnloadCoal(int minerId, int coalAmount)
        {
            warehouseSemaphore.Wait();

            try
            {
                Thread.Sleep(coalAmount * unloadTime);

                lock (warehouseLock)
                {
                    warehouseCoal += coalAmount;
                }
            }
            finally
            {
                warehouseSemaphore.Release();
            }
        }
    }
}