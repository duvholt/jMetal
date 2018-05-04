package org.uma.jmetal.experiment;

import org.uma.jmetal.algorithm.Algorithm;
import org.uma.jmetal.algorithm.multiobjective.abyss.ABYSS;
import org.uma.jmetal.algorithm.multiobjective.abyss.ABYSSBuilder;
import org.uma.jmetal.algorithm.multiobjective.mocell.MOCell;
import org.uma.jmetal.algorithm.multiobjective.mocell.MOCellBuilder;
import org.uma.jmetal.algorithm.multiobjective.moead.AbstractMOEAD;
import org.uma.jmetal.algorithm.multiobjective.moead.MOEADBuilder;
import org.uma.jmetal.algorithm.multiobjective.nsgaii.NSGAIIBuilder;
import org.uma.jmetal.algorithm.multiobjective.nsgaiii.NSGAIII;
import org.uma.jmetal.algorithm.multiobjective.nsgaiii.NSGAIIIBuilder;
import org.uma.jmetal.algorithm.multiobjective.paes.PAES;
import org.uma.jmetal.algorithm.multiobjective.paes.PAESBuilder;
import org.uma.jmetal.algorithm.multiobjective.smpso.SMPSOBuilder;
import org.uma.jmetal.algorithm.multiobjective.spea2.SPEA2Builder;
import org.uma.jmetal.operator.impl.crossover.SBXCrossover;
import org.uma.jmetal.operator.impl.mutation.PolynomialMutation;
import org.uma.jmetal.operator.impl.selection.BinaryTournamentSelection;
import org.uma.jmetal.problem.DoubleProblem;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.problem.multiobjective.UF.*;
import org.uma.jmetal.problem.multiobjective.dtlz.*;
import org.uma.jmetal.problem.multiobjective.zdt.ZDT1;
import org.uma.jmetal.problem.multiobjective.zdt.ZDT2;
import org.uma.jmetal.problem.multiobjective.zdt.ZDT3;
import org.uma.jmetal.problem.multiobjective.zdt.ZDT4;
import org.uma.jmetal.problem.multiobjective.zdt.ZDT6;
import org.uma.jmetal.qualityindicator.impl.Epsilon;
import org.uma.jmetal.qualityindicator.impl.GenerationalDistance;
import org.uma.jmetal.qualityindicator.impl.InvertedGenerationalDistance;
import org.uma.jmetal.qualityindicator.impl.InvertedGenerationalDistancePlus;
import org.uma.jmetal.qualityindicator.impl.Spread;
import org.uma.jmetal.qualityindicator.impl.hypervolume.PISAHypervolume;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.solution.Solution;
import org.uma.jmetal.util.AlgorithmRunner;
import org.uma.jmetal.util.JMetalException;
import org.uma.jmetal.util.archive.Archive;
import org.uma.jmetal.util.archive.impl.CrowdingDistanceArchive;
import org.uma.jmetal.util.comparator.RankingAndCrowdingDistanceComparator;
import org.uma.jmetal.util.evaluator.impl.SequentialSolutionListEvaluator;
import org.uma.jmetal.util.experiment.Experiment;
import org.uma.jmetal.util.experiment.ExperimentBuilder;
import org.uma.jmetal.util.experiment.component.ComputeQualityIndicators;
import org.uma.jmetal.util.experiment.component.ExecuteAlgorithms;
import org.uma.jmetal.util.experiment.component.GenerateBoxplotsWithR;
import org.uma.jmetal.util.experiment.component.GenerateFriedmanTestTables;
import org.uma.jmetal.util.experiment.component.GenerateLatexTablesWithStatistics;
import org.uma.jmetal.util.experiment.component.GenerateWilcoxonTestTablesWithR;
import org.uma.jmetal.util.experiment.util.ExperimentAlgorithm;
import org.uma.jmetal.util.experiment.util.ExperimentProblem;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Example of experimental study based on solving the ZDT problems with the algorithms NSGAII,
 * SPEA2, and SMPSO
 * <p>
 * This experiment assumes that the reference Pareto front are known, so the names of files
 * containing them and the directory where they are located must be specified.
 * <p>
 * Six quality indicators are used for performance assessment.
 * <p>
 * The steps to carry out the experiment are: 1. Configure the experiment 2. Execute the algorithms
 * 3. Compute que quality indicators 4. Generate Latex tables reporting means and medians 5.
 * Generate R scripts to produce latex tables with the result of applying the Wilcoxon Rank Sum Test
 * 6. Generate Latex tables with the ranking obtained by applying the Friedman test 7. Generate R
 * scripts to obtain boxplots
 *
 * @author Antonio J. Nebro <antonio@lcc.uma.es>
 */

public class MultiStudy {

    enum StudyType {
        ZDT,
        DTLZ,
        UF
    }

    private static final int INDEPENDENT_RUNS = 1;
    private static StudyType currentStudy = StudyType.UF;
    private static boolean enableSMPSO = false; // SMPSO is super slow

    public static void main(String[] args) throws IOException {
        String experimentBaseDirectory = "MultiStudy";

        List<ExperimentProblem<DoubleSolution>> problemList = new ArrayList<>();
        List<String> referenceFrontFileNames = null;
        if (currentStudy == StudyType.ZDT) {
            problemList.add(new ExperimentProblem<>(new ZDT1(30)));
            problemList.add(new ExperimentProblem<>(new ZDT2(30)));
            problemList.add(new ExperimentProblem<>(new ZDT3(30)));
            problemList.add(new ExperimentProblem<>(new ZDT4(30)));
            problemList.add(new ExperimentProblem<>(new ZDT6(30)));
            referenceFrontFileNames = Arrays.asList("ZDT1.pf", "ZDT2.pf", "ZDT3.pf", "ZDT4.pf", "ZDT6.pf");
        } else if (currentStudy == StudyType.DTLZ) {
            problemList.add(new ExperimentProblem<>(new DTLZ1(30, 3)));
            problemList.add(new ExperimentProblem<>(new DTLZ2(30, 3)));
            problemList.add(new ExperimentProblem<>(new DTLZ3(30, 3)));
            problemList.add(new ExperimentProblem<>(new DTLZ4(30, 3)));
            problemList.add(new ExperimentProblem<>(new DTLZ5(30, 3)));
            problemList.add(new ExperimentProblem<>(new DTLZ6(30, 3)));
            problemList.add(new ExperimentProblem<>(new DTLZ7(30, 3)));
            referenceFrontFileNames = Arrays.asList("DTLZ1.3D.pf", "DTLZ2.3D.pf", "DTLZ3.3D.pf", "DTLZ4.3D.pf", "DTLZ5.3D.pf", "DTLZ6.3D.pf", "DTLZ7.3D.pf");
        } else if (currentStudy == StudyType.UF) {
            problemList.add(new ExperimentProblem<>(new UF1(30)));
            problemList.add(new ExperimentProblem<>(new UF2(30)));
            problemList.add(new ExperimentProblem<>(new UF3(30)));
            problemList.add(new ExperimentProblem<>(new UF4(30)));
            problemList.add(new ExperimentProblem<>(new UF5(30, 10, 0.1)));
            problemList.add(new ExperimentProblem<>(new UF6(30, 2, 0.1)));
            problemList.add(new ExperimentProblem<>(new UF7(30)));
            problemList.add(new ExperimentProblem<>(new UF8(30)));
            referenceFrontFileNames = Arrays.asList("UF1.pf", "UF2.pf", "UF3.pf", "UF4.pf", "UF5.pf", "UF6.pf", "UF7.pf", "UF8.pf", "UF9.pf", "UF10.pf");
        }


        List<ExperimentAlgorithm<DoubleSolution, List<DoubleSolution>>> algorithmList =
                configureAlgorithmList(problemList);


        String experimentName = null;
        if (currentStudy == StudyType.ZDT) {
            experimentName = "zdt";
        } else if (currentStudy == StudyType.DTLZ) {
            experimentName = "dtlz";
        } else if (currentStudy == StudyType.UF) {
            experimentName = "uf";
        }

        Experiment<DoubleSolution, List<DoubleSolution>> experiment =
                new ExperimentBuilder<DoubleSolution, List<DoubleSolution>>(experimentName)
                        .setAlgorithmList(algorithmList)
                        .setProblemList(problemList)
                        .setReferenceFrontDirectory("/pareto_fronts")
                        .setReferenceFrontFileNames(referenceFrontFileNames)
                        .setExperimentBaseDirectory(experimentBaseDirectory)
                        .setOutputParetoFrontFileName("FUN")
                        .setOutputParetoSetFileName("VAR")
                        .setIndicatorList(Arrays.asList(
                                new Epsilon<DoubleSolution>(),
                                new Spread<DoubleSolution>(),
                                new GenerationalDistance<DoubleSolution>(),
                                new PISAHypervolume<DoubleSolution>(),
                                new InvertedGenerationalDistance<DoubleSolution>(),
                                new InvertedGenerationalDistancePlus<DoubleSolution>()))
                        .setIndependentRuns(INDEPENDENT_RUNS)
                        .setNumberOfCores(4)
                        .build();

        new ExecuteAlgorithms<>(experiment).run();
        new ComputeQualityIndicators<>(experiment).run();
        new GenerateLatexTablesWithStatistics(experiment).run();
        new GenerateWilcoxonTestTablesWithR<>(experiment).run();
        new GenerateFriedmanTestTables<>(experiment).run();
        new GenerateBoxplotsWithR<>(experiment).setRows(3).setColumns(3).setDisplayNotch().run();
    }

    /**
     * The algorithm list is composed of pairs {@link Algorithm} + {@link Problem} which form part of
     * a {@link ExperimentAlgorithm}, which is a decorator for class {@link Algorithm}.
     */
    static List<ExperimentAlgorithm<DoubleSolution, List<DoubleSolution>>> configureAlgorithmList(
            List<ExperimentProblem<DoubleSolution>> problemList) {
        List<ExperimentAlgorithm<DoubleSolution, List<DoubleSolution>>> algorithms = new ArrayList<>();
        // AbYSS
        for (ExperimentProblem<DoubleSolution> aProblemList1 : problemList) {
            Archive<DoubleSolution> archive = new CrowdingDistanceArchive<DoubleSolution>(100);

            ABYSS algorithm = new ABYSSBuilder((DoubleProblem) aProblemList1.getProblem(), archive)
                    .setMaxEvaluations(300000)
                    .setPopulationSize(100)
                    .build();
            algorithms.add(new ExperimentAlgorithm<>(algorithm, aProblemList1.getTag()));

        }
        // MOCell
        for (ExperimentProblem<DoubleSolution> aProblemList1 : problemList) {
            double crossoverProbability = 0.9;
            double crossoverDistributionIndex = 20.0;
            SBXCrossover crossover = new SBXCrossover(crossoverProbability, crossoverDistributionIndex);

            double mutationProbability = 1.0 / aProblemList1.getProblem().getNumberOfVariables();
            double mutationDistributionIndex = 20.0;
            PolynomialMutation mutation = new PolynomialMutation(mutationProbability, mutationDistributionIndex);

            BinaryTournamentSelection<DoubleSolution> selection = new BinaryTournamentSelection<DoubleSolution>(new RankingAndCrowdingDistanceComparator<DoubleSolution>());

            MOCell<DoubleSolution> algorithm = new MOCellBuilder<DoubleSolution>(aProblemList1.getProblem(), crossover, mutation)
                    .setSelectionOperator(selection)
                    .setMaxEvaluations(300000)
                    .setPopulationSize(100)
                    .setArchive(new CrowdingDistanceArchive<DoubleSolution>(100))
                    .build();

//            algorithms.add(new ExperimentAlgorithm<>(algorithm, aProblemList1.getTag()));
        }

        // MOEA/DD
        for (ExperimentProblem<DoubleSolution> aProblemList1 : problemList) {
            double crossoverProbability = 1.0;
            double crossoverDistributionIndex = 30.0;
            SBXCrossover crossover = new SBXCrossover(crossoverProbability, crossoverDistributionIndex);

            double mutationProbability = 1.0 / aProblemList1.getProblem().getNumberOfVariables();
            double mutationDistributionIndex = 20.0;
            PolynomialMutation mutation = new PolynomialMutation(mutationProbability, mutationDistributionIndex);

            Algorithm<List<DoubleSolution>> algorithm = new MOEADBuilder(aProblemList1.getProblem(), MOEADBuilder.Variant.MOEADD)
                    .setCrossover(crossover)
                    .setMutation(mutation)
                    .setMaxEvaluations(300000)
                    .setPopulationSize(100)
                    .setResultPopulationSize(100)
                    .setNeighborhoodSelectionProbability(0.9)
                    .setMaximumNumberOfReplacedSolutions(1)
                    .setNeighborSize(20)
                    .setFunctionType(AbstractMOEAD.FunctionType.PBI)
                    .setDataDirectory("MOEAD_Weights").build();
//            algorithms.add(new ExperimentAlgorithm<>(algorithm, aProblemList1.getTag()));
        }

        // NSGAIII
        for (ExperimentProblem<DoubleSolution> aProblemList1 : problemList) {

            double crossoverProbability = 0.9;
            double crossoverDistributionIndex = 30.0;
            SBXCrossover crossover = new SBXCrossover(crossoverProbability, crossoverDistributionIndex);

            double mutationProbability = 1.0 / aProblemList1.getProblem().getNumberOfVariables();
            double mutationDistributionIndex = 20.0;
            PolynomialMutation mutation = new PolynomialMutation(mutationProbability, mutationDistributionIndex);

            BinaryTournamentSelection<DoubleSolution> selection = new BinaryTournamentSelection<DoubleSolution>();

            NSGAIII<DoubleSolution> algorithm = new NSGAIIIBuilder<>(aProblemList1.getProblem())
                    .setCrossoverOperator(crossover)
                    .setMutationOperator(mutation)
                    .setSelectionOperator(selection)
                    .setMaxIterations(18750)
                    .setPopulationSize(100)
                    .build();
//            algorithms.add(new ExperimentAlgorithm<>(algorithm, aProblemList1.getTag()));
        }
        // NSGAII
        for (ExperimentProblem<DoubleSolution> aProblemList1 : problemList) {
            Algorithm<List<DoubleSolution>> algorithm = new NSGAIIBuilder<DoubleSolution>(
                    aProblemList1.getProblem(),
                    new SBXCrossover(1.0, 20.0),
                    new PolynomialMutation(1.0 / aProblemList1.getProblem().getNumberOfVariables(), 20.0))
                    .setMaxEvaluations(300000)
                    .setPopulationSize(100)
                    .build();
//            algorithms.add(new ExperimentAlgorithm<>(algorithm, aProblemList1.getTag()));
        }

        // PAES
        for (ExperimentProblem<DoubleSolution> aProblemList : problemList) {

            PolynomialMutation mutation = new PolynomialMutation(1.0 / aProblemList.getProblem().getNumberOfVariables(), 20.0);

            PAES<DoubleSolution> algorithm = new PAESBuilder<DoubleSolution>(aProblemList.getProblem())
                    .setMutationOperator(mutation)
                    .setMaxEvaluations(300000)
                    .setArchiveSize(100)
                    .setBiSections(5)
                    .build();
//            algorithms.add(new ExperimentAlgorithm<>(algorithm, aProblemList.getTag()));
        }

        // SMPSO
        for (ExperimentProblem<DoubleSolution> aProblemList : problemList) {
            double mutationProbability = 1.0 / aProblemList.getProblem().getNumberOfVariables();
            double mutationDistributionIndex = 20.0;
            Algorithm<List<DoubleSolution>> algorithm = new SMPSOBuilder((DoubleProblem) aProblemList.getProblem(),
                    new CrowdingDistanceArchive<DoubleSolution>(100))
                    .setMutation(new PolynomialMutation(mutationProbability, mutationDistributionIndex))
                    .setMaxIterations(3000)
                    .setSwarmSize(100)
                    .setSolutionListEvaluator(new SequentialSolutionListEvaluator<DoubleSolution>())
                    .build();
            if (enableSMPSO) {
                algorithms.add(new ExperimentAlgorithm<>(algorithm, aProblemList.getTag()));
            }
        }

        // SPEA2
        for (ExperimentProblem<DoubleSolution> aProblemList : problemList) {
            Algorithm<List<DoubleSolution>> algorithm = new SPEA2Builder<DoubleSolution>(
                    aProblemList.getProblem(),
                    new SBXCrossover(1.0, 10.0),
                    new PolynomialMutation(1.0 / aProblemList.getProblem().getNumberOfVariables(), 20.0))
                    .setMaxIterations(3000)
                    .setPopulationSize(100)
                    .build();
//            algorithms.add(new ExperimentAlgorithm<>(algorithm, aProblemList.getTag()));
        }
        return algorithms;
    }
}
