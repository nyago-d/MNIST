package mnist.util;

import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.Supplier;

/**
 *
 * @author pc02
 */
public class StreamHelper {
    
    /**
     * コンストラクタ
     */
    private StreamHelper() {
    }
    
    /**
     * ConsumerをThrowingにラップする
     */
    public static <T> Consumer<T> throwingConsumer(ThrowingConsumer<T> target) {
        return (param -> {
            try {
                target.accept(param);
            } catch (RuntimeException e) {
                throw e;
            } catch (Exception e) {
                throw new ExceptionWrapper(e);
            }
        });
    }
    
    /**
     * SupplierをThrowingにラップする
     */
    public static <T> Supplier<T> throwingSupplier(ThrowingSupplier<T> target) {
        return (() -> {
            try {
                return target.get();
            } catch (RuntimeException e) {
                throw e;
            } catch (Exception e) {
                throw new ExceptionWrapper(e);
            }
        });
    }
    
    /**
     * FunctionをThrowingにラップする
     */
    public static <T, R> Function<T, R> throwingFunction(ThrowingFunction<T, R> target) {
        return (arg -> {
            try {
                return target.apply(arg);
            } catch (RuntimeException e) {
                throw e;
            } catch (Exception e) {
                throw new ExceptionWrapper(e);
            }
        });
    }
    
    /**
     * PredicateをThrowingにラップする
     */
    public static <T> Predicate<T> throwingPredicate(ThrowingPredicate<T> target) {
        return (arg -> {
            try {
                return target.test(arg);
            } catch (RuntimeException e) {
                throw e;
            } catch (Exception e) {
                throw new ExceptionWrapper(e);
            }
        });
    }
    
    /**
     * ThrowingなConsumer
     */
    @FunctionalInterface
    public static interface ThrowingConsumer<T> {
        public void accept(T t) throws Exception;
    }

    /**
     * ThrowingなSupplier
     */    
    @FunctionalInterface
    public static interface ThrowingSupplier<T> {
        public T get() throws Exception;
    }
    
    /**
     * ThrowingなFunction
     */
    @FunctionalInterface
    public static interface ThrowingFunction<T, R> {
        public R apply(T arg) throws Exception;
    }
    
    /**
     * ThrowingなPredicate
     */
    @FunctionalInterface
    public static interface ThrowingPredicate<T> {
        public boolean test(T arg) throws Exception;
    }
    
    /**
     * ラムダの中で投げる未検査例外のWrapper
     */
    public static class ExceptionWrapper extends RuntimeException {
        private ExceptionWrapper(Throwable cause) {
            super(cause);
        }
    }

}
