#include "oofeminterface.h"
#include "../domain.h"
#include "../point.h"
#include "../vector.h"

#include "../../oofemlib/engngm.h"
#include "../../sm/EngineeringModels/DDlinearstatic.h"
#include "../../oofemlib/spatiallocalizer.h"
#include "../../oofemlib/element.h"
#include "../../oofemlib/domain.h"
#include "../../oofemlib/gausspoint.h"
#include "../../oofemlib/material.h"
#include "../../sm/Elements/structuralelement.h"
#include "../../oofemlib/generalboundarycondition.h"
#include "../../sm/Materials/linearelasticmaterial.h"
#include "../../sm/Materials/isolinearelasticmaterial.h"
#include "../../oofemlib/manualboundarycondition.h"
#include "../../oofemlib/node.h"
#include "../../oofemlib/dof.h"
#include "../../oofemlib/timestep.h"
#include "../../oofemlib/node.h"
#include "../../oofemlib/dofmanager.h"
#include "../../oofemlib/floatarray.h"
#include "../../oofemlib/dofiditem.h"
#include "../../oofemlib/bctype.h"
#include "../../oofemlib/logger.h"
#include "../../oofemlib/error.h"


namespace dd {

	void OofemInterface::addFEMContribution(const Point * point, Vector<2> &force,
                                        Vector<2> &forceGradient, Vector<3> &stress) {
    	for(int i = 1; i <= engModel->giveNumberOfDomains(); i++) {
            oofem::FloatArray localCoordinates, strainElem, stressElem;
            oofem::Element * e = engModel->giveDomain(i)->giveSpatialLocalizer()->giveElementContainingPoint(point->location());
            e->computeLocalCoordinates(localCoordinates, point->location());
               
            oofem::GaussPoint gp(e->giveDefaultIntegrationRulePtr(), -1,
                                 localCoordinates, 1, e->giveMaterialMode());
            oofem::StructuralElement * se = static_cast<oofem::StructuralElement *>(e);
            se->computeStrainVector(strainElem, &gp, engModel->giveCurrentStep());
            se->computeStressVector(stressElem, strainElem, &gp, engModel->giveCurrentStep());
                        
            force += point->getBurgersMagnitude() * point->getBurgersSign() *
                     (((stressElem[1] - stressElem[2]) * point->slipPlane()->getCos()) + 
                       stressElem[3] * point->slipPlane()->getSin() / 2);
                
        } 
    }
    
    void OofemInterface::giveNodalBcContribution(oofem::Node * node, Vector<2> &bcContribution) {
        FemInterface::giveNodalBcContribution({node->giveCoordinates()->at(1), node->giveCoordinates()->at(2)}, bcContribution);
    }
    void OofemInterface::giveNodalBcContribution(oofem::GeneralBoundaryCondition * bc, Vector<2> &bcContribution) {
        if(bc->giveType() == oofem::DirichletBT) {
            for(int dofManagerNo = 1; dofManagerNo <= bc->giveNumberOfInternalDofManagers(); dofManagerNo++) {
            	oofem::Node * node = dynamic_cast<oofem::Node *>(bc->giveInternalDofManager(dofManagerNo));
            	if(node == nullptr) { continue; }
            	giveNodalBcContribution(node, bcContribution);
            }
        }
    }
    void OofemInterface::getMaterialProperties(){
        oofem::Domain * domain = this->engModel->giveDomain(1);
        // Check here to make sure we have linear Elastic properties only
        int ddmatnum = 0;
        for ( auto &mat : domain->giveMaterials() ) {
            if (oofem::IsotropicLinearElasticMaterial * le = dynamic_cast<oofem::IsotropicLinearElasticMaterial *>(mat.get())){
                ddmatnum ++;
                if (ddmatnum > 1) {
//                     OOFEM_ERROR("Can have only one DD material currently");
                } else {
                    Domain dddomain = Domain(le->giveYoungsModulus(), le->givePoissonsRatio(), this);
                }
                
            }
        }
    }
//     void OofemInterface::applyBoundaryCondition(){
//         oofem::Domain *fem_domain = this->engModel->giveDomain(1);
//         oofem::TimeStep *tStep = this->engModel->giveCurrentStep();
//         
//         /// Loop through all DofManagers
//         for (auto &dofman : fem_domain->giveDofManagers()){
//             for (int dofid =1; dofid<= dofman->giveNumberOfDofs()){
//                 Dof *dof = dofman->giveDofWithID(dofid);
//                 if (dof->hasBc(tStep)){
//                     int bcid = dof->giveBcId();
//                     
//                     Node * node = static_cast<Node *>(dofman);
//                     giveNodalBcContribution(node, bcContribution);
//                     
//                     GeneralBoundaryCondition * bc = fem_domain->giveBc(bcid);
//                     ManualBoundaryCondition * manbc = dynamic_cast<ManualBoundaryCondition *>(fem_domain->giveBc(bcid));
//                     if(manbc == nullptr || manbc->giveType() != DirichletBT) { continue; }
//                     double toAdd;
//                     if(dof->giveDofID() == D_u) {
//                         toAdd = bcContribution[0];
//                     }
//                     else if(dof->giveDofID() == D_v) {
//                         toAdd = bcContribution[1];
//                     }
//                     else {
//                         OOFEM_ERROR("DOF must be x-disp or y-disp");
//                     }
//                     manbc->addManualValue(dof, toAdd);
//                 }
//             }
//         }
//     }


}
