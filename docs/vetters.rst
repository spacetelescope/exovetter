.. _vetters:

=======
Vetters
=======

``exovetter`` provides the following vetters.

.. automodapi:: exovetter.vetters
    :no-main-docstr:

These high-level vetter classes utilize lower level implementations laid out
in :ref:`vetters-low-level` and pass around a data structure for Threshold
Crossing Event as documented in :ref:`vetters_tce_data_structure`.

.. _vetters-low-level:

Low-Level Vetter API
====================

.. automodapi:: exovetter.lpp
    :no-inheritance-diagram:

.. automodapi:: exovetter.modshift
    :no-inheritance-diagram:

.. automodapi:: exovetter.odd_even
    :no-inheritance-diagram:

.. automodapi:: exovetter.sweet
    :no-inheritance-diagram:

.. automodapi:: exovetter.transit_coverage
    :no-inheritance-diagram:

.. _vetters_tce_data_structure:

Data Structure: Tce
===================

.. automodapi:: exovetter.tce
    :no-inheritance-diagram:
