#ifndef __POLICY_C_WRAPPER_H__
#define __POLICY_C_WRAPPER_H__

#include "SarsaAgent.h"
#include "FuncApprox.h"
#include "CMAC.h"
#include<iostream>

extern "C" {
  void* SarsaAgent_new(int numFeatures, int numActions, double learningRate, double epsilon, double lambda, void *FA, char *loadWeightsFile, char *saveWeightsFile)
  {
    CMAC *fa = reinterpret_cast<CMAC *>(FA);
    SarsaAgent *sa=new SarsaAgent(numFeatures, numActions, learningRate, epsilon, lambda, fa, loadWeightsFile, saveWeightsFile);
    void *ptr = reinterpret_cast<void *>(sa);
    return ptr;
  }
  void SarsaAgent_update(void *ptr, double state[], int action, double reward, double discountFactor)
  {
    SarsaAgent *p = reinterpret_cast<SarsaAgent *>(ptr);
    p->update(state,action,reward,discountFactor);
  }
  int SarsaAgent_selectAction(void *ptr, double state[])
  {
    SarsaAgent *p = reinterpret_cast<SarsaAgent *>(ptr);
    int action=p->selectAction(state);
    return action;
  }
  double SarsaAgent_computeStateImportance(void *ptr, double state[])
  {//compute state importance
    SarsaAgent *p = reinterpret_cast<SarsaAgent *>(ptr);
    double state_importance=p->computeStateImportance(state);
    return state_importance;
  }
  void SarsaAgent_singleSaveWeights(void *ptr)
  {// single call of saving weight...
    SarsaAgent *p = reinterpret_cast<SarsaAgent *>(ptr);
    p->singleSaveWeights();
  }
  void SarsaAgent_singleLoadWeights(void *ptr)
  {// single call of loading weight...
    SarsaAgent *p = reinterpret_cast<SarsaAgent *>(ptr);
    p->singleLoadWeights();
  }
  void SarsaAgent_endEpisode(void *ptr)
  {
    SarsaAgent *p = reinterpret_cast<SarsaAgent *>(ptr);
    p->endEpisode();
  }
}
#endif
