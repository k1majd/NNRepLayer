Running the Repair Process
=============================

Defining the constraints
-------------------------

The constraints are defined as a list of the *nnreplayer.utils.ConstraintsClass*. 

.. autoclass:: nnreplayer.utils.ConstraintsClass
    :members:

Example:
~~~~~~~~

We will go over three examples.

1. Defining "Keep Inside" Constraints:


Consider we have a linear box defined in the form Ax<b and can be visualized as shown in following table. 

.. list-table:: Constraint
    :widths: 50 50
    :header-rows: 0

    * - .. image:: constraint_1_visual.png
      - .. math:: \left[\begin{array}{cc}-0.70710678 & -0.70710678\\0.70710678 &-0.70710678\\-0.70710678 & 0.70710678\\0.70710678 & 0.70710678\end{array}\right] x \leq \left[\begin{array}{c}-2.31053391 \\1.225     \\1.225     \\4.76053391\end{array}\right]

If we want to define the such that outputs are inside the box region, this can be achieved using:

.. code-block:: python
    
    A = np.array([
            [-0.70710678, -0.70710678],
            [ 0.70710678, -0.70710678],
            [-0.70710678,  0.70710678],
            [ 0.70710678,  0.70710678]
            ])

    b = np.array([
                [-2.31053391],
                [ 1.225     ],
                [ 1.225     ],
                [ 4.76053391]
                ])

    constraint_inside = constraints_class("inside", A, b)
      