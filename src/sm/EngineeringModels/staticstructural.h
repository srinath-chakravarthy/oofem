/*
 *
 *                 #####    #####   ######  ######  ###   ###
 *               ##   ##  ##   ##  ##      ##      ## ### ##
 *              ##   ##  ##   ##  ####    ####    ##  #  ##
 *             ##   ##  ##   ##  ##      ##      ##     ##
 *            ##   ##  ##   ##  ##      ##      ##     ##
 *            #####    #####   ##      ######  ##     ##
 *
 *
 *             OOFEM : Object Oriented Finite Element Code
 *
 *               Copyright (C) 1993 - 2013   Borek Patzak
 *
 *
 *
 *       Czech Technical University, Faculty of Civil Engineering,
 *   Department of Structural Mechanics, 166 29 Prague, Czech Republic
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */

#ifndef staticstructural_h
#define staticstructural_h

#include "../sm/EngineeringModels/structengngmodel.h"
#include "../sm/EngineeringModels/xfemsolverinterface.h"
#include "sparsenonlinsystemnm.h"
#include "sparsemtrxtype.h"
#include "staggeredsolver.h"

#include <list>

#define _IFT_StaticStructural_Name "staticstructural"
#define _IFT_StaticStructural_deltat "deltat"
#define _IFT_StaticStructural_prescribedTimes "prescribedtimes"
#define _IFT_StaticStructural_solvertype "solvertype"
#define _IFT_StaticStructural_stiffmode "stiffmode"
#define _IFT_StaticStructural_nonlocalExtension "nonlocalext"

/**
 * Adaptive Time Stepping 
 * @author Srinath Chakravarthy (Northeastern University) 
 * optional parameters 
 * @par deltat_min -> Minimum deltaT allowed in this solution step
 * @par deltat_max -> Maxmimum deltaT allowed in this solution step
 * Method works based on including new solver parameters
 * Files changed 
 * 1) nrsolver.h and nrsolver.C
 * 2) nmstatus.h ---> new nrsolver parameters added
 * @todo synchronize number of solution steps to time 
 * Will not work for parallel solvers because time step in not synced between parallel domains
 * @todo synchronize deltaT between parallel domains ---> non-trivial
 */ 
#define _IFT_StaticStructural_deltat_min "deltat_min"
#define _IFT_StaticStructural_deltat_max "deltat_max"

#define _IFT_StaticStructural_recomputeaftercrackpropagation "recomputeaftercrackprop"
namespace oofem {
class SparseMtrx;

/**
 * Solves a static structural problem.
 * @author Mikael Öhman
 */
class StaticStructural : public StructuralEngngModel, public XfemSolverInterface
{
protected:
    FloatArray solution;
    FloatArray internalForces;
    FloatArray eNorm;

public:
    std :: unique_ptr< SparseMtrx >stiffnessMatrix;
protected:

    std :: unique_ptr< PrimaryField >field;

    SparseMtrxType sparseMtrxType;

    std :: unique_ptr< SparseNonLinearSystemNM >nMethod;
    int solverType;
    MatResponseMode stiffMode;

    double deltaT;
    FloatArray prescribedTimes;

    double deltaT_min, deltaT_max; ///< Max, min and starting deltaT
    
    bool adaptiveTime; ///< Flag to indicate adaptive time stepping is in use
    
    InitialGuess initialGuessType;

    bool mRecomputeStepAfterPropagation;
    
    bool converged; ///< Check for convergence of numerical method
    
    std :: list<bool>prevNSolStatus; /// < Queue containing previous n=5 converged values

public:
    StaticStructural(int i, EngngModel * _master = NULL);
    virtual ~StaticStructural();
    virtual IRResultType initializeFrom(InputRecord *ir);

    virtual void solveYourself();
    virtual void solveYourselfAt(TimeStep *tStep);

    virtual void terminate(TimeStep *tStep);

    virtual void updateComponent(TimeStep *tStep, NumericalCmpn cmpn, Domain *d);
    
    virtual double giveUnknownComponent(ValueModeType type, TimeStep *tStep, Domain *d, Dof *dof);

    virtual void updateDomainLinks();

    virtual int forceEquationNumbering();

    virtual TimeStep *giveNextStep();
    virtual double giveEndOfTimeOfInterest();
    virtual NumericalMethod *giveNumericalMethod(MetaStep *mStep);
    
    virtual fMode giveFormulation() { return TL; }

    void setSolution(TimeStep *tStep, const FloatArray &vectorToStore);

    virtual bool requiresEquationRenumbering(TimeStep *tStep);

    virtual int requiresUnknownsDictionaryUpdate() { return true; }
    virtual int giveUnknownDictHashIndx(ValueModeType mode, TimeStep *tStep);

    // identification
    virtual const char *giveInputRecordName() const { return _IFT_StaticStructural_Name; }
    virtual const char *giveClassName() const { return "StaticStructural"; }

    virtual contextIOResultType saveContext(DataStream *stream, ContextMode mode, void *obj = NULL);
    virtual contextIOResultType restoreContext(DataStream *stream, ContextMode mode, void *obj = NULL);
    
    virtual int estimateMaxPackSize(IntArray &commMap, DataStream &buff, int packUnpackType);
    
    bool proceedStep(TimeStep *tStep);
};
} // end namespace oofem
#endif // staticstructural_h
