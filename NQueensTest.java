package opt.test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Random;
import opt.ga.NQueensFitnessFunction;
import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * @author kmanda1
 * @version 1.0
 */
public class NQueensTest {
    /** The n value */
    private static final int N = 50;
    /** The t value */

    public static void write_output_to_file(String output_dir, String file_name, String results, boolean final_result) {
        try {
            if (final_result) {
                String augmented_output_dir = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date());
                String full_path = augmented_output_dir + "/" + file_name;
                Path p = Paths.get(full_path);
                if (Files.notExists(p)) {
                    Files.createDirectories(p.getParent());
                }
                PrintWriter pwtr = new PrintWriter(new BufferedWriter(new FileWriter(full_path, true)));
                synchronized (pwtr) {
                    pwtr.println(results);
                    pwtr.close();
                }
            }
            else {
                String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) + "/" + file_name;
                Path p = Paths.get(full_path);
                Files.createDirectories(p.getParent());
                Files.write(p, results.getBytes());
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void main(String[] args) {
        double start, end, time;
        int[] interations = {10,100,1000,5000,10000};
        int runs = 5;
        for (int interation: interations) {
            int sum_rhc = 0;
            int sum_sa = 0;
            int sum_ga = 0;
            int sum_mimic = 0;

            double time_rhc = 0;
            double time_sa = 0;
            double time_ga = 0;
            double time_mimic = 0;

            for (int j = 0; j < runs; j++) {
                int[] ranges = new int[N];
                Random random = new Random(N);
                for (int i = 0; i < N; i++) {
                    ranges[i] = random.nextInt();
                }
                NQueensFitnessFunction ef = new NQueensFitnessFunction();
                Distribution odd = new DiscretePermutationDistribution(N);
                NeighborFunction nf = new SwapNeighbor();
                MutationFunction mf = new SwapMutation();
                CrossoverFunction cf = new SingleCrossOver();
                Distribution df = new DiscreteDependencyTree(.1);
                HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
                GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
                ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

                start = System.nanoTime();
                RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                FixedIterationTrainer fit = new FixedIterationTrainer(rhc, interation);
                fit.train();
                end = System.nanoTime();
                time = end - start;
                time /= Math.pow(10, 9);
                sum_rhc += ef.value(rhc.getOptimal());
                time_rhc += time;
                // long starttime = System.currentTimeMillis();
                // System.out.println("RHC: " + ef.value(rhc.getOptimal()));
                // System.out.println("RHC: Board Position: ");
                // System.out.println(ef.boardPositions());
                // System.out.println("Time : " + (System.currentTimeMillis() - starttime));

                // System.out.println("============================");

                start = System.nanoTime();
                SimulatedAnnealing sa = new SimulatedAnnealing(1E1, .1, hcp);
                fit = new FixedIterationTrainer(sa, interation);
                fit.train();
                end = System.nanoTime();
                time = end - start;
                time /= Math.pow(10, 9);
                sum_sa += ef.value(sa.getOptimal());
                time_sa += time;
                // starttime = System.currentTimeMillis();
                // System.out.println("SA: " + ef.value(sa.getOptimal()));
                // System.out.println("SA: Board Position: ");
                // System.out.println(ef.boardPositions());
                // System.out.println("Time : " + (System.currentTimeMillis() - starttime));
                // System.out.println("============================");

                start = System.nanoTime();
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 0, 10, gap);
                fit = new FixedIterationTrainer(ga, interation);
                fit.train();
                end = System.nanoTime();
                time = end - start;
                time /= Math.pow(10, 9);
                sum_ga += ef.value(ga.getOptimal());
                time_ga += time;

                // System.out.println("GA: " + ef.value(ga.getOptimal()));
                // System.out.println("GA: Board Position: ");
                //System.out.println(ef.boardPositions());
                // System.out.println("Time : " + (System.currentTimeMillis() - starttime));

                // System.out.println("============================");

                start = System.nanoTime();
                MIMIC mimic = new MIMIC(200, 10, pop);
                fit = new FixedIterationTrainer(mimic, interation);
                fit.train();
                end = System.nanoTime();
                time = end - start;
                time /= Math.pow(10, 9);
                sum_mimic += ef.value(mimic.getOptimal());
                time_mimic += time;

                // System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
                // System.out.println("MIMIC: Board Position: ");
                // System.out.println(ef.boardPositions());
                // System.out.println("Time : " + (System.currentTimeMillis() - starttime));
            }
            int average_rhc = sum_rhc / runs;
            int average_sa = sum_sa / runs;
            int average_ga = sum_ga / runs;
            int average_mimic = sum_mimic / runs;

            double averagetime_rhc = time_rhc / runs;
            double averagetime_sa = time_sa / runs;
            double averagetime_ga = time_ga / runs;
            double averagetime_mimic = time_mimic / runs;

            System.out.println("##############");
            System.out.println("this is iteration " + interation);
            System.out.println("rhc average is " + average_rhc + ", time average is " + averagetime_rhc);
            System.out.println("sa average is " + average_sa + ", time average is " + averagetime_sa);
            System.out.println("ga average is " + average_ga + ", time average is " + averagetime_ga);
            System.out.println("mimic average is " + average_mimic + ", time average is " + averagetime_mimic);

            String final_result = "";
            final_result = "rhc" + "," + interation + "," + Integer.toString(average_rhc) + "," + Double.toString(averagetime_rhc) + "," +
                    "sa" + "," + interation + "," + Integer.toString(average_sa) + "," + Double.toString(averagetime_sa) + "," +
                    "ga" + "," + interation + "," + Integer.toString(average_ga) + "," + Double.toString(averagetime_ga) + "," +
                    "mimic" + "," + interation + "," + Integer.toString(average_mimic) + "," + Double.toString(averagetime_mimic);

            write_output_to_file("Optimization_Results", "nqueens_results.csv", final_result, true);

        }
    }

}
