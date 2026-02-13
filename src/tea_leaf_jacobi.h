#ifndef TEA_LEAF_JACOBI_H
#define TEA_LEAF_JACOBI_H

namespace TeaLeaf {

    /**
     * @brief Orchestre la résolution par la méthode de Jacobi sur l'ensemble des tiles.
     * * Cette fonction correspond au MODULE tea_leaf_jacobi_module du Fortran.
     * Elle boucle sur les tiles de la tâche actuelle, appelle le kernel de calcul
     * et cumule l'erreur locale avant la réduction MPI.
     * * @param error Référence pour stocker la somme des erreurs de toutes les tiles.
     */
    void tea_leaf_jacobi_solve(double& error);

} // namespace TeaLeaf

#endif // TEA_LEAF_JACOBI_H