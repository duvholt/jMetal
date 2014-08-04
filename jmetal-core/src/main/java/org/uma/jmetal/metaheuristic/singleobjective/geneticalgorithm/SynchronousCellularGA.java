//  scGA.java
//
//  Author:
//       Antonio J. Nebro <antonio@lcc.uma.es>
//       Juan J. Durillo <durillo@lcc.uma.es>
//
//  Copyright (c) 2011 Antonio J. Nebro, Juan J. Durillo
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
// 
//  You should have received a copy of the GNU Lesser General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

package org.uma.jmetal.metaheuristic.singleobjective.geneticalgorithm;

import org.uma.jmetal.core.*;
import org.uma.jmetal.util.JMetalException;
import org.uma.jmetal.util.Neighborhood;
import org.uma.jmetal.util.comparator.ObjectiveComparator;

import java.util.Comparator;
import java.util.HashMap;

/**
 * Class implementing a single-objective synchronous cellular genetic algorithm
 */
public class SynchronousCellularGA extends Algorithm {
  private static final long serialVersionUID = -7315994069313234885L;

  /** Constructor */
  public SynchronousCellularGA() {
    super();
  }


  /** execute method */
  public SolutionSet execute() throws JMetalException, ClassNotFoundException {
    int populationSize, maxEvaluations, evaluations;
    Operator mutationOperator = null;
    Operator crossoverOperator = null;
    Operator selectionOperator = null;

    SolutionSet[] neighbors;
    SolutionSet population;
    SolutionSet tempPopulation;
    Neighborhood neighborhood;

    Comparator<Solution> comparator;
    comparator = new ObjectiveComparator(0); // Single objective comparator

    HashMap<String, Object> selectionParameters = new HashMap<String, Object>();
    selectionParameters.put("comparator", comparator);
    //Operator findBestSolution = new BestSolutionSelection(selectionParameters) ;

    //Read the parameters
    populationSize = (Integer) getInputParameter("populationSize");
    maxEvaluations = (Integer) getInputParameter("maxEvaluations");

    //Read the operator
    mutationOperator = operators.get("mutation");
    crossoverOperator = operators.get("crossover");
    selectionOperator = operators.get("selection");

    //Initialize the variables    
    evaluations = 0;
    neighborhood = new Neighborhood(populationSize);
    neighbors = new SolutionSet[populationSize];

    population = new SolutionSet(populationSize);
    //Create the initial population
    for (int i = 0; i < populationSize; i++) {
      Solution solution = new Solution(problem);
      problem.evaluate(solution);
      population.add(solution);
      solution.setLocation(i);
      evaluations++;
    }

    boolean solutionFound = false;
    while ((evaluations < maxEvaluations) && !solutionFound) {
      tempPopulation = new SolutionSet(populationSize);
      for (int ind = 0; ind < population.size(); ind++) {
        Solution individual = new Solution(population.get(ind));

        Solution[] parents = new Solution[2];
        Solution[] offSpring = null;

        neighbors[ind] = neighborhood.getEightNeighbors(population, ind);
        neighbors[ind].add(individual);

        //parents
        parents[0] = (Solution) selectionOperator.execute(neighbors[ind]);
        parents[1] = (Solution) selectionOperator.execute(neighbors[ind]);

        //Create a new solutiontype, using genetic operator mutation and crossover
        if (crossoverOperator != null) {
          offSpring = (Solution[]) crossoverOperator.execute(parents);
        } else {
          offSpring = new Solution[1];
          offSpring[0] = new Solution(parents[0]);
        }
        mutationOperator.execute(offSpring[0]);

        //->Evaluate offspring and constraints
        problem.evaluate(offSpring[0]);
        evaluations++;

        if (comparator.compare(individual, offSpring[0]) < 0) {
          tempPopulation.add(individual);
        } else {
          tempPopulation.add(offSpring[0]);
        }
      }

      population = tempPopulation;
    }


    SolutionSet resultPopulation = new SolutionSet(1);
    resultPopulation.add(population.best(comparator));

    return resultPopulation;
  }
}
