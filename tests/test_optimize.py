def test_optimize_run(configure_with_daemon):
    from aiida_optimize import TestWorkChain
    print(TestWorkChain.run())
