package br.com.guialves.rflr.gymnasium4j.dto;

import br.com.guialves.rflr.gymnasium4j.utils.SocketManager;
import lombok.NoArgsConstructor;

@NoArgsConstructor
public class EnvStatus {

    public static final String REPORT = "1";
    public static final String RENDER = "2";
    public static final String METADATA_RENDER = "3";
    public static final String SAMPLE_ACTION = "4";
    public static final String DISCRETE_ACTION = "5";
    public static final String CONTINUOUS_ACTION = "6";
    public static final String DONE = "7";
}
